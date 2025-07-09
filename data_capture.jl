# data_capture.jl
using Sockets
using Logging

"""
    capture_file_data(filepath::String, output_channel::Channel{String}; 
                     close_on_finish::Bool=false, delay_ms::Int=0)

Reads a file line by line and pushes each line to the output_channel.

# Arguments
- `filepath::String`: Path to the input file
- `output_channel::Channel{String}`: Channel to send data to
- `close_on_finish::Bool`: Whether to close the channel when finished (default: false)
- `delay_ms::Int`: Delay in milliseconds between lines (default: 0, useful for simulation)
"""
function capture_file_data(filepath::String, output_channel::Channel{String}; 
                          close_on_finish::Bool=false, delay_ms::Int=0)
    @info "Starting file data capture from: $filepath"
    
    if !isfile(filepath)
        @error "File not found: $filepath"
        return false
    end
    
    lines_processed = 0
    try
        open(filepath, "r") do io
            for line in eachline(io)
                # Skip empty lines (optional behavior)
                if !isempty(strip(line))
                    put!(output_channel, line)
                    lines_processed += 1
                    
                    # Optional delay for simulation purposes
                    if delay_ms > 0
                        sleep(delay_ms / 1000.0)
                    end
                end
            end
        end
        @info "Finished file data capture from: $filepath (processed $lines_processed lines)"
        return true
    catch e
        @error "Error capturing file data from $filepath: $e"
        return false
    finally
        if close_on_finish
            close(output_channel)
            @info "Output channel closed after file capture"
        end
    end
end

"""
    capture_socket_data(port::Int, output_channel::Channel{String}; 
                       max_connections::Int=100, buffer_size::Int=1024)

Listens on a TCP port and pushes each line received from clients to the output_channel.
Handles multiple client connections concurrently.

# Arguments
- `port::Int`: TCP port to listen on
- `output_channel::Channel{String}`: Channel to send data to
- `max_connections::Int`: Maximum concurrent connections (default: 100)
- `buffer_size::Int`: Buffer size for reading (default: 1024)
"""
function capture_socket_data(port::Int, output_channel::Channel{String}; 
                           max_connections::Int=100, buffer_size::Int=1024)
    @info "Starting socket data capture on port: $port"
    
    local server_socket
    connection_count = Threads.Atomic{Int}(0)
    
    try
        server_socket = listen(Sockets.localhost, port)
        @info "Socket server listening on $(Sockets.localhost):$port"
        
        @async begin # Run server loop in a separate task
            try
                while true
                    sock = accept(server_socket)
                    
                    # Check connection limit
                    current_connections = Threads.atomic_add!(connection_count, 1)
                    if current_connections > max_connections
                        @warn "Maximum connections ($max_connections) exceeded, rejecting client $(sock.peername)"
                        close(sock)
                        Threads.atomic_sub!(connection_count, 1)
                        continue
                    end
                    
                    @info "Client connected from $(sock.peername) (active connections: $current_connections)"
                    
                    @async begin # Handle each client in its own task
                        try
                            while !eof(sock)
                                line = readline(sock; keep=false)
                                if !isempty(strip(line))
                                    put!(output_channel, line)
                                end
                            end
                            @info "Client $(sock.peername) disconnected gracefully."
                        catch e
                            if isa(e, EOFError) || isa(e, Base.IOError)
                                @info "Client $(sock.peername) disconnected."
                            else
                                @error "Error handling socket client $(sock.peername): $e"
                            end
                        finally
                            close(sock)
                            Threads.atomic_sub!(connection_count, 1)
                            @debug "Connection count decremented to $(connection_count[])"
                        end
                    end
                end
            catch e
                if !isa(e, InterruptException)
                    @error "Socket server error: $e"
                end
            end
        end
        
        return server_socket # Return handle for external control
        
    catch e
        @error "Failed to start socket server on port $port: $e"
        return nothing
    end
end

"""
    stop_socket_server(server_socket)

Gracefully stops a socket server.
"""
function stop_socket_server(server_socket)
    if server_socket !== nothing
        try
            close(server_socket)
            @info "Socket server stopped gracefully."
        catch e
            @error "Error stopping socket server: $e"
        end
    end
end

"""
    create_data_capture_system(;file_paths::Vector{String}=String[], 
                              socket_ports::Vector{Int}=Int[], 
                              channel_capacity::Int=1000)

Creates a complete data capture system with multiple sources.
Returns a tuple of (output_channel, server_handles, cleanup_function).
"""
function create_data_capture_system(;file_paths::Vector{String}=String[], 
                                   socket_ports::Vector{Int}=Int[], 
                                   channel_capacity::Int=1000)
    
    # Create shared output channel
    output_channel = Channel{String}(channel_capacity)
    server_handles = []
    tasks = []
    
    @info "Creating data capture system with $(length(file_paths)) files and $(length(socket_ports)) socket ports"
    
    # Start file capture tasks
    for filepath in file_paths
        task = @async capture_file_data(filepath, output_channel)
        push!(tasks, task)
    end
    
    # Start socket capture servers
    for port in socket_ports
        server_handle = capture_socket_data(port, output_channel)
        if server_handle !== nothing
            push!(server_handles, server_handle)
        end
    end
    
    # Cleanup function
    cleanup = function()
        @info "Shutting down data capture system..."
        
        # Stop socket servers
        for server in server_handles
            stop_socket_server(server)
        end
        
        # Wait for file tasks to complete (they should finish naturally)
        for task in tasks
            if !istaskdone(task)
                @info "Waiting for file capture task to complete..."
                wait(task)
            end
        end
        
        # Close the output channel
        close(output_channel)
        @info "Data capture system shutdown complete."
    end
    
    return (output_channel, server_handles, cleanup)
end

# Example usage patterns:

"""
Example 1: Simple file and socket capture
"""
function example_simple_usage()
    text_channel = Channel{String}(1000)
    
    # Start file capture
    @async capture_file_data("input.txt", text_channel)
    
    # Start socket capture
    server_handle = capture_socket_data(8081, text_channel)
    
    # Process data from channel
    @async begin
        for data in text_channel
            println("Received: $data")
        end
    end
    
    return text_channel, server_handle
end

"""
Example 2: Complete system with multiple sources
"""
function example_complete_system()
    # Create system with multiple file sources and socket ports
    channel, servers, cleanup = create_data_capture_system(
        file_paths=["data1.txt", "data2.txt", "logs.txt"],
        socket_ports=[8081, 8082, 8083],
        channel_capacity=5000
    )
    
    # Start data processor
    @async begin
        try
            for data in channel
                # Process your data here
                @info "Processing: $(length(data)) characters"
            end
        catch e
            if !isa(e, InvalidStateException) # Channel closed
                @error "Data processing error: $e"
            end
        end
    end
    
    # Return cleanup function for graceful shutdown
    return cleanup
end