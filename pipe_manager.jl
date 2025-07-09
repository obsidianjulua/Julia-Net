# pipe_manager.jl
using Logging

"""
    ProcessHandle

Holds references to a running external process and its associated tasks.
"""
mutable struct ProcessHandle
    process::Base.Process
    stdin_task::Task
    stdout_task::Task
    stderr_task::Union{Task, Nothing}
    command::Cmd
    start_time::Float64
    
    ProcessHandle(proc, stdin_task, stdout_task, stderr_task, cmd) = 
        new(proc, stdin_task, stdout_task, stderr_task, cmd, time())
end

"""
    process_with_external_command(
        input_channel::Channel{String},
        output_channel::Channel{String},
        command_parts::Vector{String};
        capture_stderr::Bool=false,
        stderr_channel::Union{Channel{String}, Nothing}=nothing,
        timeout_seconds::Union{Float64, Nothing}=nothing,
        close_output_on_finish::Bool=false,
        process_env::Dict{String,String}=Dict{String,String}()
    )

Reads lines from `input_channel`, pipes them as stdin to an external command,
and pushes the command's stdout to `output_channel`.

# Arguments
- `input_channel::Channel{String}`: Source of input data
- `output_channel::Channel{String}`: Destination for command output
- `command_parts::Vector{String}`: Command and arguments to execute
- `capture_stderr::Bool`: Whether to capture stderr (default: false)
- `stderr_channel::Union{Channel{String}, Nothing}`: Channel for stderr output
- `timeout_seconds::Union{Float64, Nothing}`: Process timeout (default: no timeout)
- `close_output_on_finish::Bool`: Close output channel when done (default: false)
- `process_env::Dict{String,String}`: Environment variables for the process

# Returns
- `ProcessHandle`: Handle to the running process for monitoring/control
"""
function process_with_external_command(
    input_channel::Channel{String},
    output_channel::Channel{String},
    command_parts::Vector{String};
    capture_stderr::Bool=false,
    stderr_channel::Union{Channel{String}, Nothing}=nothing,
    timeout_seconds::Union{Float64, Nothing}=nothing,
    close_output_on_finish::Bool=false,
    process_env::Dict{String,String}=Dict{String,String}()
)
    
    if isempty(command_parts)
        throw(ArgumentError("Command parts cannot be empty"))
    end
    
    # Validate stderr configuration
    if capture_stderr && stderr_channel === nothing
        @warn "capture_stderr=true but no stderr_channel provided, stderr will be logged"
    end
    
    cmd = Cmd(command_parts)
    if !isempty(process_env)
        cmd = setenv(cmd, process_env)
    end
    
    @info "Starting external command processing: $(cmd)"
    
    local proc
    local stdin_task, stdout_task, stderr_task
    
    try
        # Create process with pipes
        proc = open(cmd, "r+", stdin=true, stdout=true, stderr=capture_stderr)
        
        # Task to feed data from input_channel to the command's stdin
        stdin_task = @async begin
            lines_written = 0
            try
                for line in input_channel
                    if isopen(proc.in)
                        println(proc.in, line)
                        lines_written += 1
                    else
                        @warn "Process stdin closed, stopping input feed"
                        break
                    end
                end
                @info "Fed $lines_written lines to external command stdin"
            catch e
                if isa(e, InvalidStateException)
                    @info "Input channel closed, stopping stdin feed"
                else
                    @error "Error feeding data to external command: $e"
                end
            finally
                if isopen(proc.in)
                    close(proc.in)
                    @info "Closed stdin to external command"
                end
            end
        end
        
        # Task to read data from the command's stdout
        stdout_task = @async begin
            lines_read = 0
            try
                while !eof(proc.out) && process_running(proc)
                    line = readline(proc.out, keep=false)
                    if !isempty(line)
                        put!(output_channel, line)
                        lines_read += 1
                    end
                end
                @info "Read $lines_read lines from external command stdout"
            catch e
                if isa(e, EOFError)
                    @info "External command stdout closed"
                else
                    @error "Error reading data from external command stdout: $e"
                end
            finally
                if isopen(proc.out)
                    close(proc.out)
                    @info "Closed stdout from external command"
                end
                if close_output_on_finish
                    close(output_channel)
                    @info "Closed output channel"
                end
            end
        end
        
        # Task to handle stderr if requested
        stderr_task = if capture_stderr
            @async begin
                lines_read = 0
                try
                    while !eof(proc.err) && process_running(proc)
                        line = readline(proc.err, keep=false)
                        if !isempty(line)
                            if stderr_channel !== nothing
                                put!(stderr_channel, line)
                            else
                                @warn "STDERR: $line"
                            end
                            lines_read += 1
                        end
                    end
                    @info "Read $lines_read lines from external command stderr"
                catch e
                    if isa(e, EOFError)
                        @info "External command stderr closed"
                    else
                        @error "Error reading stderr from external command: $e"
                    end
                finally
                    if isopen(proc.err)
                        close(proc.err)
                    end
                end
            end
        else
            nothing
        end
        
        # Create process handle
        handle = ProcessHandle(proc, stdin_task, stdout_task, stderr_task, cmd)
        
        # Start monitoring task
        @async monitor_process(handle, timeout_seconds)
        
        return handle
        
    catch e
        @error "Failed to execute external command $(cmd): $e"
        # Cleanup on failure
        if @isdefined(proc) && process_running(proc)
            kill(proc)
        end
        rethrow(e)
    end
end

"""
    monitor_process(handle::ProcessHandle, timeout_seconds::Union{Float64, Nothing})

Monitors a process for completion or timeout.
"""
function monitor_process(handle::ProcessHandle, timeout_seconds::Union{Float64, Nothing})
    try
        if timeout_seconds !== nothing
            # Wait with timeout
            start_time = time()
            while process_running(handle.process) && (time() - start_time) < timeout_seconds
                sleep(0.1)
            end
            
            if process_running(handle.process)
                @warn "Process $(handle.command) timed out after $timeout_seconds seconds, killing"
                kill(handle.process)
                wait(handle.process)
            end
        else
            # Wait indefinitely
            wait(handle.process)
        end
        
        runtime = time() - handle.start_time
        @info "External command $(handle.command) finished with exit code: $(handle.process.exitcode) (runtime: $(round(runtime, digits=2))s)"
        
    catch e
        @error "Error monitoring process $(handle.command): $e"
    end
end

"""
    is_process_running(handle::ProcessHandle)

Check if a process is still running.
"""
function is_process_running(handle::ProcessHandle)
    return process_running(handle.process)
end

"""
    kill_process(handle::ProcessHandle; force::Bool=false)

Kill a running process gracefully or forcefully.
"""
function kill_process(handle::ProcessHandle; force::Bool=false)
    if process_running(handle.process)
        if force
            @info "Force killing process $(handle.command)"
            kill(handle.process, Base.SIGKILL)
        else
            @info "Gracefully terminating process $(handle.command)"
            kill(handle.process, Base.SIGTERM)
        end
        return true
    else
        @info "Process $(handle.command) is not running"
        return false
    end
end

"""
    wait_for_process(handle::ProcessHandle; timeout_seconds::Union{Float64, Nothing}=nothing)

Wait for a process to complete, optionally with timeout.
"""
function wait_for_process(handle::ProcessHandle; timeout_seconds::Union{Float64, Nothing}=nothing)
    if timeout_seconds !== nothing
        start_time = time()
        while process_running(handle.process) && (time() - start_time) < timeout_seconds
            sleep(0.1)
        end
        
        if process_running(handle.process)
            @warn "Timeout waiting for process $(handle.command)"
            return false
        end
    else
        wait(handle.process)
    end
    
    # Wait for associated tasks to complete
    if !istaskdone(handle.stdin_task)
        wait(handle.stdin_task)
    end
    if !istaskdone(handle.stdout_task)
        wait(handle.stdout_task)
    end
    if handle.stderr_task !== nothing && !istaskdone(handle.stderr_task)
        wait(handle.stderr_task)
    end
    
    return true
end

"""
    create_pipeline(stages::Vector{Vector{String}}, 
                   input_channel::Channel{String};
                   intermediate_buffer_size::Int=1000,
                   capture_stderr::Bool=false,
                   timeout_seconds::Union{Float64, Nothing}=nothing)

Creates a pipeline of external commands, chaining their inputs and outputs.

# Arguments
- `stages::Vector{Vector{String}}`: Vector of command parts for each stage
- `input_channel::Channel{String}`: Initial input channel
- `intermediate_buffer_size::Int`: Size of intermediate channels (default: 1000)
- `capture_stderr::Bool`: Whether to capture stderr from all stages
- `timeout_seconds::Union{Float64, Nothing}`: Timeout for each stage

# Returns
- `Tuple{Channel{String}, Vector{ProcessHandle}}`: Final output channel and process handles
"""
function create_pipeline(stages::Vector{Vector{String}}, 
                        input_channel::Channel{String};
                        intermediate_buffer_size::Int=1000,
                        capture_stderr::Bool=false,
                        timeout_seconds::Union{Float64, Nothing}=nothing)
    
    if isempty(stages)
        throw(ArgumentError("Pipeline stages cannot be empty"))
    end
    
    @info "Creating pipeline with $(length(stages)) stages"
    
    channels = [input_channel]
    handles = ProcessHandle[]
    
    # Create intermediate channels
    for i in 1:(length(stages)-1)
        push!(channels, Channel{String}(intermediate_buffer_size))
    end
    
    # Create final output channel
    final_output = Channel{String}(intermediate_buffer_size)
    push!(channels, final_output)
    
    # Start each stage
    for (i, stage_cmd) in enumerate(stages)
        @info "Starting pipeline stage $i: $(stage_cmd)"
        
        is_last_stage = (i == length(stages))
        
        handle = process_with_external_command(
            channels[i],
            channels[i+1],
            stage_cmd;
            capture_stderr=capture_stderr,
            timeout_seconds=timeout_seconds,
            close_output_on_finish=is_last_stage
        )
        
        push!(handles, handle)
    end
    
    return final_output, handles
end

# Example usage patterns:

"""
Example 1: Simple grep command
"""
function example_grep()
    input_chan = Channel{String}(100)
    output_chan = Channel{String}(100)
    
    # Start grep process
    handle = process_with_external_command(
        input_chan, output_chan, ["grep", "ERROR"];
        capture_stderr=true,
        timeout_seconds=30.0
    )
    
    # Feed some data
    @async begin
        for i in 1:10
            put!(input_chan, "Log line $i: $(i % 3 == 0 ? "ERROR" : "INFO") message")
        end
        close(input_chan)
    end
    
    # Read results
    @async begin
        for line in output_chan
            println("Filtered: $line")
        end
    end
    
    return handle
end

"""
Example 2: Multi-stage pipeline (grep -> sort -> uniq)
"""
function example_pipeline()
    input_chan = Channel{String}(100)
    
    # Create pipeline: grep -> sort -> uniq
    output_chan, handles = create_pipeline(
        [["grep", "ERROR"], ["sort"], ["uniq", "-c"]],
        input_chan;
        intermediate_buffer_size=500,
        capture_stderr=true
    )
    
    # Feed data
    @async begin
        logs = ["ERROR: Connection failed", "INFO: Starting", "ERROR: Timeout", 
                "ERROR: Connection failed", "INFO: Success", "ERROR: Timeout"]
        for log in logs
            put!(input_chan, log)
        end
        close(input_chan)
    end
    
    # Process results
    @async begin
        for line in output_chan
            println("Pipeline result: $line")
        end
    end
    
    return handles
end