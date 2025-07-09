# integrated_personal_data_system.jl
"""
Integrated Personal Data Server with Replicode Training Integration

Combines data capture, processing, storage, and advanced behavioral training
with Replicode-based cognitive modeling.
"""

using Dates
using JSON3
using SQLite
using Logging
using UUIDs
using Statistics

# Include all our modules
include("data_capture.jl")
include("pipe_manager.jl") 
include("encoding_pipeline.jl")
include("personal_data_server.jl")

# Include Replicode training modules (from your files)
include("imp1.txt")  # This contains your ReplicodeTrainingRegiment and BehavioralTrainingRegiment
using .ReplicodeTrainingRegiment
using .BehavioralTrainingRegiment

export IntegratedPersonalDataSystem, DataIntelligenceEngine, CognitiveProcessor
export create_intelligent_data_system, start_cognitive_processing, analyze_behavioral_patterns

"""
    CognitiveProcessor

Integrates Replicode training with real-time data processing.
"""
mutable struct CognitiveProcessor
    replicode_config::ReplicodeConfig
    training_orchestrator::TrainingOrchestrator
    behavioral_trainer::AutomatedTrainer
    intent_classifier::IntentClassifier
    
    # Processing state
    active_learning::Bool
    cognitive_cycles::Int
    behavioral_adaptations::Int
    
    # Performance tracking
    intent_accuracy::Float64
    adaptation_success_rate::Float64
    cognitive_load::Float64
    
    function CognitiveProcessor(config::ReplicodeConfig)
        orchestrator = TrainingOrchestrator()
        orchestrator.global_config = config
        
        trainer = AutomatedTrainer()
        classifier = create_intent_classifier()
        
        new(config, orchestrator, trainer, classifier,
            false, 0, 0, 0.0, 0.0, 0.0)
    end
end

"""
    DataIntelligenceEngine

Core engine that combines data processing with cognitive modeling.
"""
mutable struct DataIntelligenceEngine
    # Data processing components
    data_server::PersonalDataServer
    cognitive_processor::CognitiveProcessor
    
    # Intelligence features
    pattern_recognition::Dict{String, Any}
    behavioral_profiles::Dict{String, BehavioralProfile}
    intent_history::Vector{Dict{String, Any}}
    
    # Real-time processing
    intelligence_channels::Dict{String, Channel}
    processing_tasks::Vector{Task}
    
    # Analytics
    intelligence_stats::Dict{String, Any}
    
    function DataIntelligenceEngine(config_file::String="intelligent_config.json")
        # Load configuration
        config = load_intelligent_config(config_file)
        
        # Create data server
        data_server = PersonalDataServer(config["data_server_config"])
        
        # Create cognitive processor with Replicode config
        replicode_config = create_replicode_config_from_dict(config["replicode_config"])
        cognitive_processor = CognitiveProcessor(replicode_config)
        
        new(data_server, cognitive_processor, 
            Dict{String, Any}(), Dict{String, BehavioralProfile}(), 
            Dict{String, Any}[], Dict{String, Channel}(), Task[],
            Dict{String, Any}())
    end
end

"""
    IntegratedPersonalDataSystem

Main system class that orchestrates everything.
"""
mutable struct IntegratedPersonalDataSystem
    intelligence_engine::DataIntelligenceEngine
    api_server::Union{APIServer, Nothing}
    
    # System state
    system_active::Bool
    startup_time::DateTime
    
    # Monitoring
    health_monitor::Task
    performance_tracker::Dict{String, Any}
    
    function IntegratedPersonalDataSystem(config_file::String="system_config.json")
        engine = DataIntelligenceEngine(config_file)
        
        new(engine, nothing, false, now(), 
            Task(() -> nothing), Dict{String, Any}())
    end
end

"""
    load_intelligent_config(config_file::String)

Load comprehensive configuration for the intelligent system.
"""
function load_intelligent_config(config_file::String)
    if isfile(config_file)
        return JSON3.read(read(config_file, String), Dict{String, Any})
    else
        # Default intelligent configuration
        default_config = Dict{String, Any}(
            "data_server_config" => "data_server.json",
            "replicode_config" => Dict{String, Any}(
                "runtime_ms" => 30000,
                "probe_level" => 2,
                "base_period" => 50.0,
                "reduction_core_count" => 4,
                "time_core_count" => 2,
                "model_inertia_success_rate" => 0.85,
                "model_inertia_count" => 8,
                "tpx_delta_success_rate" => 0.80,
                "min_sim_time_horizon" => 300.0,
                "max_sim_time_horizon" => 1200.0,
                "tpx_time_horizon" => 600.0,
                "perf_sampling_period" => 200.0,
                "float_tolerance" => 0.0001,
                "timer_tolerance" => 15.0,
                "notif_marker_resilience" => 0.90,
                "goal_pred_success_resilience" => 0.85,
                "debug_enabled" => true,
                "debug_windows" => false,
                "trace_levels" => Dict(
                    "composite_inputs" => false,
                    "composite_outputs" => false,
                    "model_inputs" => true,
                    "model_outputs" => true,
                    "prediction_monitoring" => true,
                    "goal_monitoring" => true,
                    "model_revision" => true
                )
            ),
            "intelligence_config" => Dict{String, Any}(
                "enable_real_time_learning" => true,
                "behavioral_adaptation_rate" => 0.1,
                "intent_confidence_threshold" => 0.75,
                "pattern_recognition_depth" => 5,
                "cognitive_load_threshold" => 0.8,
                "auto_profile_creation" => true,
                "cross_user_learning" => false
            ),
            "api_config" => Dict{String, Any}(
                "port" => 8080,
                "enable_websockets" => true,
                "cors_enabled" => true,
                "rate_limiting" => Dict(
                    "requests_per_minute" => 1000,
                    "burst_size" => 100
                )
            )
        )
        
        # Save default config
        open(config_file, "w") do f
            JSON3.pretty(f, default_config)
        end
        
        @info "Created default intelligent system configuration: $config_file"
        return default_config
    end
end

"""
    create_replicode_config_from_dict(config_dict::Dict{String, Any})

Convert configuration dictionary to ReplicodeConfig.
"""
function create_replicode_config_from_dict(config_dict::Dict{String, Any})
    config = ReplicodeConfig()
    
    # Map configuration values
    config.runtime_ms = config_dict["runtime_ms"]
    config.probe_level = config_dict["probe_level"]
    config.base_period = config_dict["base_period"]
    config.reduction_core_count = config_dict["reduction_core_count"]
    config.time_core_count = config_dict["time_core_count"]
    config.model_inertia_success_rate = config_dict["model_inertia_success_rate"]
    config.model_inertia_count = config_dict["model_inertia_count"]
    config.tpx_delta_success_rate = config_dict["tpx_delta_success_rate"]
    config.min_sim_time_horizon = config_dict["min_sim_time_horizon"]
    config.max_sim_time_horizon = config_dict["max_sim_time_horizon"]
    config.tpx_time_horizon = config_dict["tpx_time_horizon"]
    config.perf_sampling_period = config_dict["perf_sampling_period"]
    config.float_tolerance = config_dict["float_tolerance"]
    config.timer_tolerance = config_dict["timer_tolerance"]
    config.notif_marker_resilience = config_dict["notif_marker_resilience"]
    config.goal_pred_success_resilience = config_dict["goal_pred_success_resilience"]
    config.debug_enabled = config_dict["debug_enabled"]
    config.debug_windows = config_dict["debug_windows"]
    config.trace_levels = config_dict["trace_levels"]
    
    return config
end

"""
    create_intelligent_data_system(config_file::String="system_config.json")

Create and configure the complete intelligent data system.
"""
function create_intelligent_data_system(config_file::String="system_config.json")
    @info "Creating Integrated Personal Data System with Cognitive Processing..."
    
    system = IntegratedPersonalDataSystem(config_file)
    
    # Initialize intelligence channels
    setup_intelligence_channels!(system.intelligence_engine)
    
    # Configure data sources with intelligence integration
    setup_intelligent_data_sources!(system.intelligence_engine)
    
    @info "Intelligent data system created successfully"
    return system
end

"""
    setup_intelligence_channels!(engine::DataIntelligenceEngine)

Set up communication channels for intelligent processing.
"""
function setup_intelligence_channels!(engine::DataIntelligenceEngine)
    # Create intelligence processing channels
    engine.intelligence_channels["raw_data"] = Channel{String}(2000)
    engine.intelligence_channels["processed_data"] = Channel{String}(2000)
    engine.intelligence_channels["behavioral_data"] = Channel{Dict{String, Any}}(1000)
    engine.intelligence_channels["intent_predictions"] = Channel{Dict{String, Any}}(1000)
    engine.intelligence_channels["cognitive_feedback"] = Channel{Dict{String, Any}}(500)
    
    @info "Intelligence processing channels established"
end

"""
    setup_intelligent_data_sources!(engine::DataIntelligenceEngine)

Configure data sources with cognitive processing integration.
"""
function setup_intelligent_data_sources!(engine::DataIntelligenceEngine)
    # Add intelligent log processing source
    log_source_config = Dict{String, Any}(
        "name" => "Intelligent Log Processor",
        "source_type" => "file",
        "config" => Dict{String, Any}("filepath" => "logs/user_behavior.log"),
        "processors" => [
            ["grep", "-E", "(user_action|intent|behavior)"],
            ["jq", "-c", "."]  # Parse JSON logs
        ],
        "encoder_config" => Dict{String, Any}(
            "type" => "tfidf",
            "strategy" => "positional",
            "embedding_dim" => 256,
            "vocab_size" => 5000
        ),
        "intelligence_enabled" => true
    )
    
    log_source_id = add_data_source(engine.data_server, log_source_config)
    
    # Add real-time behavioral data source
    behavior_source_config = Dict{String, Any}(
        "name" => "Real-time Behavioral Stream",
        "source_type" => "socket",
        "config" => Dict{String, Any}("port" => 8081),
        "encoder_config" => Dict{String, Any}(
            "type" => "dummy",
            "strategy" => "bag_of_words",
            "embedding_dim" => 128
        ),
        "intelligence_enabled" => true
    )
    
    behavior_source_id = add_data_source(engine.data_server, behavior_source_config)
    
    @info "Intelligent data sources configured: $log_source_id, $behavior_source_id"
end

"""
    start_cognitive_processing(system::IntegratedPersonalDataSystem)

Start the complete system with cognitive processing.
"""
function start_cognitive_processing(system::IntegratedPersonalDataSystem)
    @info "Starting Integrated Personal Data System with Cognitive Processing..."
    
    system.system_active = true
    system.startup_time = now()
    
    # Start data server
    start_server(system.intelligence_engine.data_server)
    
    # Start cognitive processor
    start_cognitive_engine!(system.intelligence_engine)
    
    # Start intelligence processing tasks
    start_intelligence_tasks!(system.intelligence_engine)
    
    # Start health monitoring
    system.health_monitor = @async monitor_system_health(system)
    
    # Start API server if configured
    config = load_intelligent_config()
    if get(config, "api_enabled", true)
        system.api_server = APIServer(system.intelligence_engine.data_server, 
                                     config["api_config"]["port"])
        start_api_server(system.api_server)
    end
    
    @info "Integrated Personal Data System is now running with cognitive capabilities"
    return true
end

"""
    start_cognitive_engine!(engine::DataIntelligenceEngine)

Initialize and start the cognitive processing engine.
"""
function start_cognitive_engine!(engine::DataIntelligenceEngine)
    @info "Starting cognitive processing engine..."
    
    # Start Replicode training orchestrator
    engine.cognitive_processor.active_learning = true
    
    # Create training pipeline for real-time learning
    pipeline_config = engine.cognitive_processor.replicode_config
    real_time_pipeline = create_training_pipeline(pipeline_config, "real_time_intelligence")
    engine.cognitive_processor.training_orchestrator.pipelines["real_time_intelligence"] = real_time_pipeline
    
    @info "Cognitive engine started with Replicode training pipeline"
end

"""
    start_intelligence_tasks!(engine::DataIntelligenceEngine)

Start background tasks for intelligent processing.
"""
function start_intelligence_tasks!(engine::DataIntelligenceEngine)
    # Task 1: Process raw data into behavioral insights
    push!(engine.processing_tasks, @async process_behavioral_insights(engine))
    
    # Task 2: Real-time intent classification
    push!(engine.processing_tasks, @async classify_user_intents(engine))
    
    # Task 3: Adaptive learning from feedback
    push!(engine.processing_tasks, @async adaptive_learning_processor(engine))
    
    # Task 4: Pattern recognition and profile evolution
    push!(engine.processing_tasks, @async evolve_behavioral_profiles(engine))
    
    # Task 5: Cognitive load monitoring and optimization
    push!(engine.processing_tasks, @async monitor_cognitive_load(engine))
    
    @info "Started $(length(engine.processing_tasks)) intelligence processing tasks"
end

"""
    process_behavioral_insights(engine::DataIntelligenceEngine)

Process raw data to extract behavioral insights.
"""
function process_behavioral_insights(engine::DataIntelligenceEngine)
    @info "Starting behavioral insights processor..."
    
    processed_count = 0
    
    # Connect to data server's processed text channel
    for (source_id, source) in engine.data_server.data_sources
        if haskey(engine.data_server.server_channels, "$(source_id)_processed")
            processed_channel = engine.data_server.server_channels["$(source_id)_processed"]
            
            @async begin
                for processed_text in processed_channel
                    try
                        # Extract behavioral features
                        behavioral_data = extract_behavioral_data(processed_text, source_id)
                        
                        # Send to behavioral processing channel
                        put!(engine.intelligence_channels["behavioral_data"], behavioral_data)
                        
                        processed_count += 1
                        if processed_count % 100 == 0
                            @info "Processed $processed_count behavioral insights"
                        end
                        
                    catch e
                        @error "Error processing behavioral insight: $e"
                    end
                end
            end
        end
    end
end

"""
    extract_behavioral_data(text::String, source_id::String)

Extract behavioral patterns from processed text.
"""
function extract_behavioral_data(text::String, source_id::String)
    # Parse text for behavioral indicators
    behavioral_data = Dict{String, Any}(
        "timestamp" => now(),
        "source_id" => source_id,
        "raw_text" => text,
        "extracted_features" => Dict{String, Any}()
    )
    
    # Extract various behavioral features
    features = behavioral_data["extracted_features"]
    
    # Text sentiment and complexity
    features["text_length"] = length(text)
    features["word_count"] = length(split(text))
    features["complexity_score"] = estimate_text_complexity(text)
    
    # Intent indicators
    intent_keywords = Dict(
        "help" => ["help", "assist", "support", "guide"],
        "create" => ["create", "make", "build", "generate"],
        "analyze" => ["analyze", "examine", "study", "investigate"],
        "learn" => ["learn", "understand", "explain", "teach"]
    )
    
    for (intent, keywords) in intent_keywords
        features["intent_$(intent)_score"] = count_keyword_matches(text, keywords) / length(keywords)
    end
    
    # Temporal features
    features["hour_of_day"] = Dates.hour(now())
    features["day_of_week"] = Dates.dayofweek(now())
    
    # Context extraction (if JSON structured)
    try
        if startswith(text, "{") && endswith(text, "}")
            parsed_data = JSON3.read(text, Dict{String, Any})
            features["structured_data"] = parsed_data
        end
    catch
        # Not JSON, continue with text processing
    end
    
    return behavioral_data
end

"""
    estimate_text_complexity(text::String)

Estimate the complexity of text content.
"""
function estimate_text_complexity(text::String)
    words = split(text)
    if isempty(words)
        return 0.0
    end
    
    # Simple complexity metrics
    avg_word_length = mean(length.(words))
    unique_word_ratio = length(unique(words)) / length(words)
    sentence_count = count(c -> c in ['.', '!', '?'], text)
    avg_sentence_length = length(words) / max(1, sentence_count)
    
    # Combine metrics
    complexity = (avg_word_length / 10.0) + unique_word_ratio + (avg_sentence_length / 20.0)
    return min(1.0, complexity)
end

"""
    count_keyword_matches(text::String, keywords::Vector{String})

Count how many keywords appear in the text.
"""
function count_keyword_matches(text::String, keywords::Vector{String})
    lower_text = lowercase(text)
    matches = 0
    for keyword in keywords
        if contains(lower_text, lowercase(keyword))
            matches += 1
        end
    end
    return matches
end

"""
    classify_user_intents(engine::DataIntelligenceEngine)

Real-time intent classification using behavioral data.
"""
function classify_user_intents(engine::DataIntelligenceEngine)
    @info "Starting real-time intent classifier..."
    
    classified_count = 0
    
    for behavioral_data in engine.intelligence_channels["behavioral_data"]
        try
            # Extract features for intent classification
            features = create_feature_vector_from_behavioral_data(behavioral_data)
            
            # Use cognitive processor for intent prediction
            predicted_intent, confidence = predict_intent(
                engine.cognitive_processor.intent_classifier, features)
            
            # Create intent prediction record
            intent_prediction = Dict{String, Any}(
                "timestamp" => now(),
                "source_id" => behavioral_data["source_id"],
                "predicted_intent" => predicted_intent,
                "confidence" => confidence,
                "features" => behavioral_data["extracted_features"],
                "behavioral_data_id" => hash(behavioral_data["raw_text"])
            )
            
            # Send to intent predictions channel
            put!(engine.intelligence_channels["intent_predictions"], intent_prediction)
            
            # Store in intent history
            push!(engine.intent_history, intent_prediction)
            
            # Keep intent history manageable
            if length(engine.intent_history) > 10000
                engine.intent_history = engine.intent_history[end-5000:end]
            end
            
            classified_count += 1
            if classified_count % 50 == 0
                @info "Classified $classified_count intents (recent: $predicted_intent @ $(round(confidence, digits=3)))"
            end
            
        catch e
            @error "Error in intent classification: $e"
        end
    end
end

"""
    create_feature_vector_from_behavioral_data(behavioral_data::Dict{String, Any})

Convert behavioral data to feature vector for ML processing.
"""
function create_feature_vector_from_behavioral_data(behavioral_data::Dict{String, Any})
    features = zeros(Float32, 256)
    
    extracted = get(behavioral_data, "extracted_features", Dict{String, Any}())
    
    # Text features
    features[1] = get(extracted, "text_length", 0.0) / 1000.0  # Normalize
    features[2] = get(extracted, "word_count", 0.0) / 100.0
    features[3] = get(extracted, "complexity_score", 0.0)
    
    # Intent scores
    features[4] = get(extracted, "intent_help_score", 0.0)
    features[5] = get(extracted, "intent_create_score", 0.0)
    features[6] = get(extracted, "intent_analyze_score", 0.0)
    features[7] = get(extracted, "intent_learn_score", 0.0)
    
    # Temporal features
    features[8] = get(extracted, "hour_of_day", 0.0) / 24.0
    features[9] = get(extracted, "day_of_week", 0.0) / 7.0
    
    # Hash text for unique signature
    text_hash = hash(get(behavioral_data, "raw_text", ""))
    for i in 10:20
        features[i] = (text_hash % (i * 1000)) / (i * 1000.0)
    end
    
    return features
end

"""
    adaptive_learning_processor(engine::DataIntelligenceEngine)

Process feedback and adapt learning parameters.
"""
function adaptive_learning_processor(engine::DataIntelligenceEngine)
    @info "Starting adaptive learning processor..."
    
    adaptation_count = 0
    
    for feedback_data in engine.intelligence_channels["cognitive_feedback"]
        try
            # Extract user ID and feedback
            user_id = get(feedback_data, "user_id", "anonymous")
            feedback_score = get(feedback_data, "feedback_score", 0.0)
            intent_context = get(feedback_data, "intent_context", Dict{String, Any}())
            
            # Use behavioral trainer for adaptive learning
            action = get(intent_context, "action", "unknown_action")
            context = get(intent_context, "context", Dict{String, Any}())
            
            result = adaptive_learning_loop(
                engine.cognitive_processor.behavioral_trainer,
                user_id,
                action,
                context,
                feedback_score
            )
            
            # Update cognitive processor stats
            engine.cognitive_processor.cognitive_cycles += 1
            if feedback_score > 0.5
                engine.cognitive_processor.behavioral_adaptations += 1
            end
            
            adaptation_count += 1
            if adaptation_count % 10 == 0
                @info "Processed $adaptation_count adaptive learning cycles"
            end
            
        catch e
            @error "Error in adaptive learning: $e"
        end
    end
end

"""
    evolve_behavioral_profiles(engine::DataIntelligenceEngine)

Evolve and update behavioral profiles based on new data.
"""
function evolve_behavioral_profiles(engine::DataIntelligenceEngine)
    @info "Starting behavioral profile evolution..."
    
    while engine.cognitive_processor.active_learning
        try
            # Analyze recent intent predictions for patterns
            if length(engine.intent_history) >= 20
                recent_intents = engine.intent_history[end-19:end]
                
                # Group by source or user context
                intent_patterns = analyze_intent_patterns(recent_intents)
                
                # Update or create behavioral profiles
                for (pattern_id, pattern_data) in intent_patterns
                    update_behavioral_profile!(engine, pattern_id, pattern_data)
                end
            end
            
            # Sleep between evolution cycles
            sleep(30)  # Evolve every 30 seconds
            
        catch e
            @error "Error in profile evolution: $e"
            sleep(60)  # Wait longer on error
        end
    end
end

"""
    analyze_intent_patterns(intent_history::Vector{Dict{String, Any}})

Analyze patterns in intent history.
"""
function analyze_intent_patterns(intent_history::Vector{Dict{String, Any}})
    patterns = Dict{String, Dict{String, Any}}()
    
    # Group by source_id
    by_source = Dict{String, Vector{Dict{String, Any}}}()
    for intent in intent_history
        source_id = get(intent, "source_id", "unknown")
        if !haskey(by_source, source_id)
            by_source[source_id] = []
        end
        push!(by_source[source_id], intent)
    end
    
    # Analyze each source's patterns
    for (source_id, intents) in by_source
        if length(intents) >= 3
            intent_sequence = [intent["predicted_intent"] for intent in intents]
            avg_confidence = mean([intent["confidence"] for intent in intents])
            
            patterns[source_id] = Dict{String, Any}(
                "intent_sequence" => intent_sequence,
                "average_confidence" => avg_confidence,
                "most_common_intent" => mode(intent_sequence),
                "intent_diversity" => length(unique(intent_sequence)) / length(intent_sequence),
                "sample_count" => length(intents)
            )
        end
    end
    
    return patterns
end

"""
    update_behavioral_profile!(engine::DataIntelligenceEngine, 
                              profile_id::String, pattern_data::Dict{String, Any})

Update or create a behavioral profile based on patterns.
"""
function update_behavioral_profile!(engine::DataIntelligenceEngine, 
                                   profile_id::String, pattern_data::Dict{String, Any})
    if !haskey(engine.behavioral_profiles, profile_id)
        # Create new profile
        engine.behavioral_profiles[profile_id] = BehavioralProfile(profile_id)
    end
    
    profile = engine.behavioral_profiles[profile_id]
    
    # Add variables based on detected patterns
    most_common_intent = pattern_data["most_common_intent"]
    confidence = pattern_data["average_confidence"]
    diversity = pattern_data["intent_diversity"]
    
    # Add intent preference variable
    intent_var_name = "preferred_intent_$(most_common_intent)"
    add_variable(profile, intent_var_name, "intent_preference", 
                confidence, "profile_evolution", "behavioral_analysis")
    
    # Add diversity variable
    diversity_var_name = "intent_diversity"
    add_variable(profile, diversity_var_name, "behavioral_trait",
                diversity, "profile_evolution", "behavioral_analysis")
    
    @debug "Updated behavioral profile $profile_id with $(length(profile.variables)) variables"
end

"""
    monitor_cognitive_load(engine::DataIntelligenceEngine)

Monitor and optimize cognitive processing load.
"""
function monitor_cognitive_load(engine::DataIntelligenceEngine)
    @info "Starting cognitive load monitor..."
    
    while engine.cognitive_processor.active_learning
        try
            # Calculate current cognitive load
            load_metrics = calculate_cognitive_load(engine)
            engine.cognitive_processor.cognitive_load = load_metrics["total_load"]
            
            # Optimize if load is too high
            if load_metrics["total_load"] > 0.8
                optimize_cognitive_processing!(engine, load_metrics)
            end
            
            # Update performance stats
            update_cognitive_performance_stats!(engine, load_metrics)
            
            sleep(60)  # Monitor every minute
            
        catch e
            @error "Error in cognitive load monitoring: $e"
            sleep(120)
        end
    end
end

"""
    calculate_cognitive_load(engine::DataIntelligenceEngine)

Calculate current cognitive processing load.
"""
function calculate_cognitive_load(engine::DataIntelligenceEngine)
    load_metrics = Dict{String, Float64}()
    
    # Channel load (percentage of capacity used)
    channel_loads = []
    for (name, channel) in engine.intelligence_channels
        if isa(channel, Channel)
            capacity = channel.sz_max
            current_size = length(channel.data)
            load = capacity > 0 ? current_size / capacity : 0.0
            push!(channel_loads, load)
        end
    end
    
    load_metrics["channel_load"] = isempty(channel_loads) ? 0.0 : mean(channel_loads)
    
    # Processing task load
    active_tasks = count(task -> !istaskdone(task), engine.processing_tasks)
    load_metrics["task_load"] = active_tasks / max(1, length(engine.processing_tasks))
    
    # Profile complexity load
    total_variables = sum(length(profile.variables) for profile in values(engine.behavioral_profiles))
    load_metrics["profile_complexity_load"] = min(1.0, total_variables / 1000.0)
    
    # Intent history load
    load_metrics["intent_history_load"] = min(1.0, length(engine.intent_history) / 10000.0)
    
    # Calculate total load
    load_metrics["total_load"] = mean([
        load_metrics["channel_load"],
        load_metrics["task_load"], 
        load_metrics["profile_complexity_load"],
        load_metrics["intent_history_load"]
    ])
    
    return load_metrics
end

"""
    optimize_cognitive_processing!(engine::DataIntelligenceEngine, load_metrics::Dict{String, Float64})

Optimize cognitive processing when load is high.
"""
function optimize_cognitive_processing!(engine::DataIntelligenceEngine, load_metrics::Dict{String, Float64})
    @info "Optimizing cognitive processing (load: $(round(load_metrics["total_load"], digits=3)))"
    
    # Reduce intent history if too large
    if load_metrics["intent_history_load"] > 0.7
        target_size = Int(length(engine.intent_history) * 0.7)
        engine.intent_history = engine.intent_history[end-target_size:end]
        @info "Reduced intent history to $target_size entries"
    end
    
    # Simplify behavioral profiles if too complex
    if load_metrics["profile_complexity