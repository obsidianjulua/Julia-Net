# personal_data_server.jl
"""
Personal Data Server and Management Framework

A comprehensive system for ingesting, processing, storing, and analyzing personal data
with advanced tokenization and text processing capabilities.
"""

using Dates
using JSON3
using SQLite
using Logging
using UUIDs

# Include our existing modules
include("data_capture.jl")
include("pipe_manager.jl") 
include("encoding_pipeline.jl")

"""
    DataSource

Represents a configured data source with its processing pipeline.
"""
struct DataSource
    id::String
    name::String
    source_type::Symbol  # :file, :socket, :api, :database
    config::Dict{String, Any}
    processors::Vector{Vector{String}}  # External command pipelines
    encoder_config::Dict{String, Any}
    created_at::DateTime
    enabled::Bool
end

"""
    ProcessedDocument

Represents a processed document with metadata and embeddings.
"""
struct ProcessedDocument
    id::String
    source_id::String
    original_text::String
    processed_text::String
    embedding::Vector{Float64}
    metadata::Dict{String, Any}
    created_at::DateTime
    content_hash::String
end

"""
    PersonalDataServer

Main server class that orchestrates data ingestion, processing, and storage.
"""
mutable struct PersonalDataServer
    config::Dict{String, Any}
    data_sources::Dict{String, DataSource}
    database::SQLite.DB
    encoders::Dict{String, AbstractTextEncoder}
    active_captures::Dict{String, Any}  # Running capture processes
    active_processors::Dict{String, Any}  # Running processing pipelines
    server_channels::Dict{String, Channel}
    stats::Dict{String, Any}
    started_at::DateTime
    
    function PersonalDataServer(config_file::String="config.json")
        config = load_config(config_file)
        db = setup_database(config["database_path"])
        
        new(
            config,
            Dict{String, DataSource}(),
            db,
            Dict{String, AbstractTextEncoder}(),
            Dict{String, Any}(),
            Dict{String, Any}(),
            Dict{String, Channel}(),
            Dict{String, Any}(),
            now()
        )
    end
end

"""
    load_config(config_file::String)

Load server configuration from JSON file.
"""
function load_config(config_file::String)
    if isfile(config_file)
        return JSON3.read(read(config_file, String), Dict{String, Any})
    else
        # Default configuration
        default_config = Dict{String, Any}(
            "database_path" => "personal_data.db",
            "max_concurrent_sources" => 10,
            "default_embedding_dim" => 256,
            "default_vocab_size" => 10000,
            "max_document_size" => 1000000,  # 1MB
            "cleanup_interval_hours" => 24,
            "backup_interval_hours" => 6,
            "log_level" => "INFO",
            "server_port" => 8080,
            "api_enabled" => true,
            "web_interface_enabled" => true
        )
        
        # Save default config
        open(config_file, "w") do f
            JSON3.pretty(f, default_config)
        end
        
        @info "Created default configuration file: $config_file"
        return default_config
    end
end

"""
    setup_database(db_path::String)

Initialize SQLite database with required tables.
"""
function setup_database(db_path::String)
    db = SQLite.DB(db_path)
    
    # Create tables
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS data_sources (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            config TEXT NOT NULL,
            processors TEXT,
            encoder_config TEXT,
            created_at TEXT NOT NULL,
            enabled BOOLEAN DEFAULT 1
        )
    """)
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            original_text TEXT NOT NULL,
            processed_text TEXT,
            embedding BLOB,
            metadata TEXT,
            created_at TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            FOREIGN KEY (source_id) REFERENCES data_sources (id)
        )
    """)
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS processing_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            documents_processed INTEGER,
            total_tokens INTEGER,
            processing_time_ms INTEGER,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (source_id) REFERENCES data_sources (id)
        )
    """)
    
    SQLite.execute(db, """
        CREATE INDEX IF NOT EXISTS idx_documents_source_id ON documents(source_id)
    """)
    
    SQLite.execute(db, """
        CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)
    """)
    
    SQLite.execute(db, """
        CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)
    """)
    
    @info "Database initialized: $db_path"
    return db
end

"""
    add_data_source(server::PersonalDataServer, source_config::Dict{String, Any})

Add a new data source to the server.
"""
function add_data_source(server::PersonalDataServer, source_config::Dict{String, Any})
    source_id = string(uuid4())
    
    # Validate required fields
    required_fields = ["name", "source_type"]
    for field in required_fields
        if !haskey(source_config, field)
            throw(ArgumentError("Missing required field: $field"))
        end
    end
    
    # Create data source
    source = DataSource(
        source_id,
        source_config["name"],
        Symbol(source_config["source_type"]),
        get(source_config, "config", Dict{String, Any}()),
        get(source_config, "processors", Vector{String}[]),
        get(source_config, "encoder_config", Dict{String, Any}()),
        now(),
        get(source_config, "enabled", true)
    )
    
    # Save to database
    SQLite.execute(server.database, """
        INSERT INTO data_sources 
        (id, name, source_type, config, processors, encoder_config, created_at, enabled)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        source.id,
        source.name,
        string(source.source_type),
        JSON3.write(source.config),
        JSON3.write(source.processors),
        JSON3.write(source.encoder_config),
        string(source.created_at),
        source.enabled
    ])
    
    # Store in memory
    server.data_sources[source_id] = source
    
    @info "Added data source: $(source.name) (ID: $source_id)"
    return source_id
end

"""
    create_encoder_for_source(server::PersonalDataServer, source::DataSource)

Create and configure an encoder for a data source.
"""
function create_encoder_for_source(server::PersonalDataServer, source::DataSource)
    encoder_config = source.encoder_config
    
    encoder_type = Symbol(get(encoder_config, "type", "dummy"))
    vocab_size = get(encoder_config, "vocab_size", server.config["default_vocab_size"])
    embedding_dim = get(encoder_config, "embedding_dim", server.config["default_embedding_dim"])
    encoding_strategy = Symbol(get(encoder_config, "strategy", "bag_of_words"))
    
    encoder = create_encoding_pipeline(
        encoder_type,
        vocab_size,
        embedding_dim;
        encoding_strategy=encoding_strategy
    )
    
    encoder_key = "$(source.id)_encoder"
    server.encoders[encoder_key] = encoder
    
    @info "Created encoder for source $(source.name): $encoder_type with $embedding_dim dimensions"
    return encoder_key
end

"""
    start_data_source(server::PersonalDataServer, source_id::String)

Start processing data from a specific source.
"""
function start_data_source(server::PersonalDataServer, source_id::String)
    if !haskey(server.data_sources, source_id)
        throw(ArgumentError("Data source not found: $source_id"))
    end
    
    source = server.data_sources[source_id]
    if !source.enabled
        @warn "Data source $(source.name) is disabled"
        return false
    end
    
    @info "Starting data source: $(source.name)"
    
    # Create processing channels
    raw_channel = Channel{String}(1000)
    processed_channel = Channel{String}(1000)
    embedding_channel = Channel{Vector{Float64}}(1000)
    
    # Store channels for cleanup
    server.server_channels["$(source_id)_raw"] = raw_channel
    server.server_channels["$(source_id)_processed"] = processed_channel
    server.server_channels["$(source_id)_embedding"] = embedding_channel
    
    # Start data capture based on source type
    capture_handle = if source.source_type == :file
        filepath = source.config["filepath"]
        @async capture_file_data(filepath, raw_channel; close_on_finish=true)
    elseif source.source_type == :socket
        port = source.config["port"]
        capture_socket_data(port, raw_channel)
    else
        throw(ArgumentError("Unsupported source type: $(source.source_type)"))
    end
    
    server.active_captures[source_id] = capture_handle
    
    # Start processing pipeline if configured
    if !isempty(source.processors)
        pipeline_output, pipeline_handles = create_pipeline(
            source.processors,
            raw_channel;
            intermediate_buffer_size=500
        )
        server.active_processors[source_id] = pipeline_handles
        processed_input = pipeline_output
    else
        processed_input = raw_channel
    end
    
    # Create and start encoder
    encoder_key = create_encoder_for_source(server, source)
    encoder = server.encoders[encoder_key]
    
    @async process_for_embedding(
        processed_input,
        embedding_channel,
        encoder;
        batch_size=10,
        normalize_vectors=true,
        close_output_on_finish=false
    )
    
    # Start document storage
    @async store_processed_documents(server, source_id, processed_input, embedding_channel)
    
    @info "Data source $(source.name) started successfully"
    return true
end

"""
    store_processed_documents(server::PersonalDataServer, source_id::String, 
                            text_channel::Channel{String}, 
                            embedding_channel::Channel{Vector{Float64}})

Store processed documents with embeddings to database.
"""
function store_processed_documents(server::PersonalDataServer, source_id::String,
                                 text_channel::Channel{String},
                                 embedding_channel::Channel{Vector{Float64}})
    @info "Starting document storage for source: $source_id"
    
    processed_count = 0
    text_buffer = String[]
    embedding_buffer = Vector{Float64}[]
    
    # Buffer texts and embeddings for batch processing
    text_task = @async begin
        for text in text_channel
            push!(text_buffer, text)
        end
    end
    
    embedding_task = @async begin
        for embedding in embedding_channel
            push!(embedding_buffer, embedding)
        end
    end
    
    # Process when both are available
    @async begin
        while !istaskdone(text_task) || !istaskdone(embedding_task) || 
              !isempty(text_buffer) || !isempty(embedding_buffer)
            
            if !isempty(text_buffer) && !isempty(embedding_buffer)
                text = popfirst!(text_buffer)
                embedding = popfirst!(embedding_buffer)
                
                # Create document
                doc = ProcessedDocument(
                    string(uuid4()),
                    source_id,
                    text,
                    text,  # processed_text same as original for now
                    embedding,
                    Dict{String, Any}(),
                    now(),
                    string(hash(text))
                )
                
                # Store in database
                store_document(server, doc)
                processed_count += 1
                
                if processed_count % 100 == 0
                    @info "Stored $processed_count documents for source $source_id"
                end
            else
                sleep(0.1)  # Wait for more data
            end
        end
        
        @info "Finished storing documents for source $source_id. Total: $processed_count"
    end
end

"""
    store_document(server::PersonalDataServer, doc::ProcessedDocument)

Store a single document in the database.
"""
function store_document(server::PersonalDataServer, doc::ProcessedDocument)
    # Check for duplicates
    existing = SQLite.Query(server.database, 
        "SELECT id FROM documents WHERE content_hash = ?", 
        [doc.content_hash]) |> collect
    
    if !isempty(existing)
        @debug "Document already exists, skipping: $(doc.content_hash)"
        return false
    end
    
    # Serialize embedding as bytes
    embedding_bytes = reinterpret(UInt8, doc.embedding)
    
    SQLite.execute(server.database, """
        INSERT INTO documents 
        (id, source_id, original_text, processed_text, embedding, metadata, created_at, content_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        doc.id,
        doc.source_id,
        doc.original_text,
        doc.processed_text,
        embedding_bytes,
        JSON3.write(doc.metadata),
        string(doc.created_at),
        doc.content_hash
    ])
    
    return true
end

"""
    start_server(server::PersonalDataServer)

Start the personal data server and all configured data sources.
"""
function start_server(server::PersonalDataServer)
    @info "Starting Personal Data Server..."
    
    # Load existing data sources from database
    sources = SQLite.Query(server.database, "SELECT * FROM data_sources WHERE enabled = 1") |> collect
    
    for row in sources
        source = DataSource(
            row.id,
            row.name,
            Symbol(row.source_type),
            JSON3.read(row.config, Dict{String, Any}),
            JSON3.read(row.processors, Vector{Vector{String}}),
            JSON3.read(row.encoder_config, Dict{String, Any}),
            DateTime(row.created_at),
            Bool(row.enabled)
        )
        
        server.data_sources[row.id] = source
        
        # Start the source
        try
            start_data_source(server, row.id)
        catch e
            @error "Failed to start data source $(source.name): $e"
        end
    end
    
    @info "Personal Data Server started with $(length(server.data_sources)) data sources"
    return true
end

"""
    stop_server(server::PersonalDataServer)

Gracefully stop the server and all processing.
"""
function stop_server(server::PersonalDataServer)
    @info "Stopping Personal Data Server..."
    
    # Stop all captures
    for (source_id, handle) in server.active_captures
        try
            if handle isa Base.Process
                kill_process(handle)
            end
        catch e
            @error "Error stopping capture for $source_id: $e"
        end
    end
    
    # Stop all processors
    for (source_id, handles) in server.active_processors
        for handle in handles
            try
                kill_process(handle)
            catch e
                @error "Error stopping processor for $source_id: $e"
            end
        end
    end
    
    # Close all channels
    for (name, channel) in server.server_channels
        try
            close(channel)
        catch e
            @debug "Channel $name already closed"
        end
    end
    
    # Close database
    SQLite.close(server.database)
    
    @info "Personal Data Server stopped"
end

"""
    get_server_stats(server::PersonalDataServer)

Get comprehensive server statistics.
"""
function get_server_stats(server::PersonalDataServer)
    # Document counts by source
    doc_counts = SQLite.Query(server.database, """
        SELECT source_id, COUNT(*) as count 
        FROM documents 
        GROUP BY source_id
    """) |> collect
    
    # Total documents
    total_docs = SQLite.Query(server.database, "SELECT COUNT(*) as count FROM documents") |> collect
    total_count = isempty(total_docs) ? 0 : total_docs[1].count
    
    # Server runtime
    runtime = now() - server.started_at
    
    stats = Dict{String, Any}(
        "server_runtime" => string(runtime),
        "total_documents" => total_count,
        "active_sources" => length(server.active_captures),
        "configured_sources" => length(server.data_sources),
        "document_counts_by_source" => Dict(row.source_id => row.count for row in doc_counts),
        "encoder_stats" => Dict(
            key => encoder.stats for (key, encoder) in server.encoders
        )
    )
    
    return stats
end

# Example usage and configuration

"""
Example: Setting up a complete personal data server
"""
function example_setup_server()
    # Create server
    server = PersonalDataServer("personal_data_config.json")
    
    # Add file-based data source
    file_source_config = Dict{String, Any}(
        "name" => "Personal Logs",
        "source_type" => "file",
        "config" => Dict{String, Any}("filepath" => "logs/personal.log"),
        "processors" => [["grep", "-v", "DEBUG"], ["sed", "s/ERROR/CRITICAL/g"]],
        "encoder_config" => Dict{String, Any}(
            "type" => "dummy",
            "strategy" => "tfidf",
            "embedding_dim" => 128,
            "vocab_size" => 5000
        )
    )
    
    file_source_id = add_data_source(server, file_source_config)
    
    # Add socket-based data source
    socket_source_config = Dict{String, Any}(
        "name" => "Real-time Messages",
        "source_type" => "socket",
        "config" => Dict{String, Any}("port" => 8081),
        "encoder_config" => Dict{String, Any}(
            "type" => "dummy",
            "strategy" => "positional",
            "embedding_dim" => 256
        )
    )
    
    socket_source_id = add_data_source(server, socket_source_config)
    
    # Start server
    start_server(server)
    
    return server
end