# api_server.jl
"""
REST API and Management Interface for Personal Data Server
"""

using HTTP
using JSON3
using Dates

"""
    APIServer

HTTP API server for managing the personal data server.
"""
mutable struct APIServer
    data_server::PersonalDataServer
    http_server::Union{HTTP.Server, Nothing}
    port::Int
    
    APIServer(data_server::PersonalDataServer, port::Int=8080) = new(data_server, nothing, port)
end

"""
    start_api_server(api_server::APIServer)

Start the HTTP API server.
"""
function start_api_server(api_server::APIServer)
    @info "Starting API server on port $(api_server.port)"
    
    # Define routes
    router = HTTP.Router()
    
    # Health check
    HTTP.register!(router, "GET", "/health", health_check)
    
    # Server management
    HTTP.register!(router, "GET", "/api/stats", req -> get_stats_handler(req, api_server.data_server))
    HTTP.register!(router, "POST", "/api/restart", req -> restart_server_handler(req, api_server.data_server))
    
    # Data source management
    HTTP.register!(router, "GET", "/api/sources", req -> list_sources_handler(req, api_server.data_server))
    HTTP.register!(router, "POST", "/api/sources", req -> create_source_handler(req, api_server.data_server))
    HTTP.register!(router, "GET", "/api/sources/*", req -> get_source_handler(req, api_server.data_server))
    HTTP.register!(router, "PUT", "/api/sources/*", req -> update_source_handler(req, api_server.data_server))
    HTTP.register!(router, "DELETE", "/api/sources/*", req -> delete_source_handler(req, api_server.data_server))
    HTTP.register!(router, "POST", "/api/sources/*/start", req -> start_source_handler(req, api_server.data_server))
    HTTP.register!(router, "POST", "/api/sources/*/stop", req -> stop_source_handler(req, api_server.data_server))
    
    # Document management
    HTTP.register!(router, "GET", "/api/documents", req -> list_documents_handler(req, api_server.data_server))
    HTTP.register!(router, "GET", "/api/documents/*", req -> get_document_handler(req, api_server.data_server))
    HTTP.register!(router, "POST", "/api/search", req -> search_documents_handler(req, api_server.data_server))
    
    # Web interface
    HTTP.register!(router, "GET", "/", serve_web_interface)
    HTTP.register!(router, "GET", "/dashboard", serve_dashboard)
    
    # CORS middleware
    cors_middleware = HTTP.Middleware.CORSMiddleware(
        origins=["*"],
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        headers=["Content-Type", "Authorization"]
    )
    
    # Start server
    api_server.http_server = HTTP.serve!(router, "0.0.0.0", api_server.port; middleware=[cors_middleware])
    
    @info "API server started on http://localhost:$(api_server.port)"
    return api_server.http_server
end

"""
    stop_api_server(api_server::APIServer)

Stop the HTTP API server.
"""
function stop_api_server(api_server::APIServer)
    if api_server.http_server !== nothing
        HTTP.close(api_server.http_server)
        api_server.http_server = nothing
        @info "API server stopped"
    end
end

# API Handlers

function health_check(req::HTTP.Request)
    return HTTP.Response(200, JSON3.write(Dict("status" => "healthy", "timestamp" => now())))
end

function get_stats_handler(req::HTTP.Request, server::PersonalDataServer)
    try
        stats = get_server_stats(server)
        return HTTP.Response(200, JSON3.write(stats))
    catch e
        @error "Error getting stats: $e"
        return HTTP.Response(500, JSON3.write(Dict("error" => "Internal server error")))
    end
end

function list_sources_handler(req::HTTP.Request, server::PersonalDataServer)
    try
        sources = []
        for (id, source) in server.data_sources
            push!(sources, Dict(
                "id" => id,
                "name" => source.name,
                "type" => string(source.source_type),
                "enabled" => source.enabled,
                "created_at" => string(source.created_at)
            ))
        end
        
        return HTTP.Response(200, JSON3.write(Dict("sources" => sources)))
    catch e
        @error "Error listing sources: $e"
        return HTTP.Response(500, JSON3.write(Dict("error" => "Internal server error")))
    end
end

function create_source_handler(req::HTTP.Request, server::PersonalDataServer)
    try
        body = JSON3.read(String(req.body), Dict{String, Any})
        source_id = add_data_source(server, body)
        
        return HTTP.Response(201, JSON3.write(Dict(
            "id" => source_id,
            "message" => "Data source created successfully"
        )))
    catch e
        @error "Error creating source: $e"
        return HTTP.Response(400, JSON3.write(Dict("error" => string(e))))
    end
end

function start_source_handler(req::HTTP.Request, server::PersonalDataServer)
    try
        source_id = extract_id_from_path(req.target)
        success = start_data_source(server, source_id)
        
        if success
            return HTTP.Response(200, JSON3.write(Dict("message" => "Source started successfully")))
        else
            return HTTP.Response(400, JSON3.write(Dict("error" => "Failed to start source")))
        end
    catch e
        @error "Error starting source: $e"
        return HTTP.Response(400, JSON3.write(Dict("error" => string(e))))
    end
end

function list_documents_handler(req::HTTP.Request, server::PersonalDataServer)
    try
        # Parse query parameters
        query_params = HTTP.URIs.queryparams(HTTP.URI(req.target).query)
        limit = parse(Int, get(query_params, "limit", "50"))
        offset = parse(Int, get(query_params, "offset", "0"))
        source_id = get(query_params, "source_id", nothing)
        
        # Build query
        sql = "SELECT id, source_id, LEFT(original_text, 200) as preview, created_at, content_hash FROM documents"
        params = []
        
        if source_id !== nothing
            sql *= " WHERE source_id = ?"
            push!(params, source_id)
        end
        
        sql *= " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        push!(params, limit, offset)
        
        # Execute query
        docs = SQLite.Query(server.database, sql, params) |> collect
        
        documents = []
        for doc in docs
            push!(documents, Dict(
                "id" => doc.id,
                "source_id" => doc.source_id,
                "preview" => doc.preview,
                "created_at" => doc.created_at,
                "content_hash" => doc.content_hash
            ))
        end
        
        return HTTP.Response(200, JSON3.write(Dict("documents" => documents)))
    catch e
        @error "Error listing documents: $e"
        return HTTP.Response(500, JSON3.write(Dict("error" => "Internal server error")))
    end
end

function search_documents_handler(req::HTTP.Request, server::PersonalDataServer)
    try
        body = JSON3.read(String(req.body), Dict{String, Any})
        query_text = body["query"]
        limit = get(body, "limit", 10)
        
        # Simple text search for now (could be enhanced with semantic search using embeddings)
        sql = """
            SELECT id, source_id, original_text, created_at, content_hash 
            FROM documents 
            WHERE original_text LIKE ? 
            ORDER BY created_at DESC 
            LIMIT ?
        """
        
        docs = SQLite.Query(server.database, sql, ["%$query_text%", limit]) |> collect
        
        results = []
        for doc in docs
            push!(results, Dict(
                "id" => doc.id,
                "source_id" => doc.source_id,
                "text" => doc.original_text,
                "created_at" => doc.created_at,
                "relevance_score" => 1.0  # Placeholder
            ))
        end
        
        return HTTP.Response(200, JSON3.write(Dict(
            "query" => query_text,
            "results" => results,
            "total_found" => length(results)
        )))
    catch e
        @error "Error searching documents: $e"
        return HTTP.Response(500, JSON3.write(Dict("error" => "Internal server error")))
    end
end

function extract_id_from_path(path::String)
    parts = split(path, "/")
    for i in 1:length(parts)
        if parts[i] == "sources" && i < length(parts)
            return parts[i+1]
        end
    end
    throw(ArgumentError("Invalid path: $path"))
end

# Web Interface Handlers

function serve_web_interface(req::HTTP.Request)
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Personal Data Server</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #333; }
            .nav { display: flex; gap: 20px; margin: 20px 0; }
            .nav a { padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }
            .nav a:hover { background: #0056b3; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
            .stat-item { text-align: center; }
            .stat-value { font-size: 2em; font-weight: bold; color: #007bff; }
            .stat-label { color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">