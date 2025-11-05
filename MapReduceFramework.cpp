#include "MapReduceFramework.h"
#include "MapReduceClient.h"
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <queue>
#include <condition_variable>
#include <iostream>
#include <unordered_map>

// ============ Barrier ===============
class Barrier {
public:
    explicit Barrier(int numThreads) : count(0), generation(0), numThreads(numThreads) {}
    ~Barrier() = default;
    void barrier();

private:
    std::mutex mutex_;
    std::condition_variable cv;
    int count;
    int generation;
    const int numThreads;
};

void Barrier::barrier() {
    std::unique_lock<std::mutex> lock(mutex_);
    int gen = generation;

    if (++count < numThreads) {
        cv.wait(lock, [this, gen] { return gen != generation; });
    } else {
        count = 0;
        generation++;
        cv.notify_all();
    }
}

//========================================

// Forward declaration
struct JobContext;

// ============ ThreadContext ===============
struct ThreadContext {
    int thread_id;
    std::vector<std::pair<K2*, V2*>> intermediate_vec;
    JobContext* job_ctx;  // Reference to job context
    
    explicit ThreadContext(int id, JobContext* jctx) : thread_id(id), job_ctx(jctx) {}
};

// ============ JobContext ===============
struct JobContext {
    const MapReduceClient& client;
    const std::vector<std::pair<K1*, V1*>>& input_vec;
    std::vector<std::pair<K3*, V3*>>& output_vec;
    std::vector<std::thread> threads;
    std::vector<ThreadContext*> thread_contexts;
    
    // Synchronization
    Barrier* barrier;
    std::mutex output_mutex;
    std::mutex shuffled_vectors_mutex;
    std::condition_variable shuffle_cv;
    
    // Atomic counters and state - using single 64-bit atomic as required
    std::atomic<uint64_t> map_counter{0};
    std::atomic<uint64_t> state_atomic{0}; // 2 bits stage, 31 bits processed, 31 bits total
    std::atomic<uint64_t> intermediate_counter{0}; // Count intermediate pairs
    std::atomic<uint64_t> shuffled_counter{0}; // Count how many vectors in shuffle queue
    std::atomic<bool> shuffle_finished{false};
    
    // Shuffled data - vector of vectors as specified in instructions
    std::vector<std::vector<std::pair<K2*, V2*>>> shuffled_vectors;
    std::atomic<uint64_t> next_vector_index{0}; // For reduce phase work distribution
    
    // For job state management
    std::atomic<bool> threads_joined{false};
    std::atomic<bool> job_finished{false};
    
    JobContext(const MapReduceClient& c, const std::vector<std::pair<K1*, V1*>>& input, 
               std::vector<std::pair<K3*, V3*>>& output, int multi_thread_level)
        : client(c), input_vec(input), output_vec(output),
          barrier(new Barrier(multi_thread_level)) {
        
        // Initialize state with MAP stage and input size as total
        uint64_t total_inputs = input_vec.size();
        uint64_t initial_state = (static_cast<uint64_t>(MAP_STAGE)) | 
                                ((total_inputs & 0x7FFFFFFFULL) << 33);
        state_atomic.store(initial_state);
        
        // Pre-allocate thread contexts
        thread_contexts.reserve(multi_thread_level);
        for (int i = 0; i < multi_thread_level; ++i) {
            thread_contexts.push_back(new ThreadContext(i, this));
        }
    }
    
    ~JobContext() {
        // Clean up thread contexts
        for (ThreadContext* ctx : thread_contexts) {
            delete ctx;
        }
        delete barrier;
    }
    
    void setStage(stage_t stage) {
        uint64_t processed = 0; // Reset processed when changing stage
        uint64_t total = 0;
        
        // Set new total based on stage
        if (stage == MAP_STAGE) {
            total = input_vec.size();
        } else if (stage == SHUFFLE_STAGE) {
            total = intermediate_counter.load();
        } else if (stage == REDUCE_STAGE) {
            total = shuffled_counter.load();
        }
        
        uint64_t new_state = (static_cast<uint64_t>(stage)) | 
                            ((processed & 0x7FFFFFFFULL) << 2) | 
                            ((total & 0x7FFFFFFFULL) << 33);
        state_atomic.store(new_state);
    }
    
    void updateProgress(uint64_t processed_count) {
        while (true) {
            uint64_t current = state_atomic.load();
            stage_t stage = static_cast<stage_t>(current & 3);
            uint64_t old_processed = (current >> 2) & 0x7FFFFFFFULL;
            uint64_t total = (current >> 33) & 0x7FFFFFFFULL;
            
            // Ensure progress is monotonic and bounded
            uint64_t new_processed = std::max(old_processed, std::min(processed_count, total));
            
            uint64_t new_state = (static_cast<uint64_t>(stage)) | 
                                ((new_processed & 0x7FFFFFFFULL) << 2) | 
                                ((total & 0x7FFFFFFFULL) << 33);
            
            // Use compare_and_swap to avoid race conditions
            if (state_atomic.compare_exchange_weak(current, new_state)) {
                break;
            }
        }
    }
    
    void getState(JobState* state) {
        uint64_t current = state_atomic.load();
        state->stage = static_cast<stage_t>(current & 3);
        uint64_t processed = (current >> 2) & 0x7FFFFFFFULL;
        uint64_t total = (current >> 33) & 0x7FFFFFFFULL;
        
        if (total == 0) {
            state->percentage = 100.0f; // If no work, we're done
        } else {
            state->percentage = (float)processed / (float)total * 100.0f;
        }
    }
};

//=========================================

// =============GLOBAL=================
std::mutex global_mutex;
std::unordered_map<JobHandle, JobContext*> job_map;
//======================================

//============ Phases ===================

// MAP PHASE
void mapPhase(JobContext* job_ctx, ThreadContext* thread_ctx) {
    while (true) {
        uint64_t old_value = job_ctx->map_counter.fetch_add(1);
        if (old_value >= job_ctx->input_vec.size()) {
            break;
        }
        
        auto& pair = job_ctx->input_vec[old_value];
        job_ctx->client.map(pair.first, pair.second, thread_ctx);
        
        // Update progress - use the incremented counter value
        job_ctx->updateProgress(old_value + 1);
    }
}

// SORT PHASE
void sortPhase(ThreadContext* thread_ctx) {
    std::sort(thread_ctx->intermediate_vec.begin(), thread_ctx->intermediate_vec.end(),
              [](const std::pair<K2*, V2*>& a, const std::pair<K2*, V2*>& b) {
                  return *a.first < *b.first;
              });
}

// SHUFFLE PHASE - Following the instructions exactly
void shufflePhase(JobContext* job_ctx) {
    job_ctx->setStage(SHUFFLE_STAGE);
    
    // Create vectors that will hold elements by key
    std::vector<std::vector<std::pair<K2*, V2*>>> key_vectors;
    std::vector<K2*> keys_seen;
    
    uint64_t total_intermediate = job_ctx->intermediate_counter.load();
    uint64_t processed_pairs = 0;
    
    // Process each thread's sorted intermediate vector
    for (const ThreadContext* ctx : job_ctx->thread_contexts) {
        for (const auto& pair : ctx->intermediate_vec) {
            // Find if we already have a vector for this key
            bool found = false;
            for (size_t i = 0; i < keys_seen.size(); ++i) {
                // Check key equality using < operator as specified
                if (!(*keys_seen[i] < *pair.first) && !(*pair.first < *keys_seen[i])) {
                    key_vectors[i].push_back(pair);
                    found = true;
                    break;
                }
            }
            
            // If key not found, create new vector for this key
            if (!found) {
                keys_seen.push_back(pair.first);
                key_vectors.push_back(std::vector<std::pair<K2*, V2*>>());
                key_vectors.back().push_back(pair);
            }
            
            // Update shuffle progress periodically to avoid too many atomic operations
            processed_pairs++;
            if (processed_pairs % 100 == 0 || processed_pairs == total_intermediate) {
                job_ctx->updateProgress(processed_pairs);
            }
        }
    }
    
    // Ensure we report 100% completion for shuffle
    if (total_intermediate > 0) {
        job_ctx->updateProgress(total_intermediate);
    }
    
    // CRITICAL FIX: Set up shuffled data and counters BEFORE marking as finished
    {
        std::lock_guard<std::mutex> lock(job_ctx->shuffled_vectors_mutex);
        job_ctx->shuffled_vectors = std::move(key_vectors);
        job_ctx->shuffled_counter.store(job_ctx->shuffled_vectors.size());
        
        // CRITICAL: Set stage and mark shuffle finished while holding lock
        job_ctx->setStage(REDUCE_STAGE);
        job_ctx->shuffle_finished.store(true);
    }
    
    // Notify all waiting reduce threads
    job_ctx->shuffle_cv.notify_all();
}

// REDUCE PHASE
void reducePhase(JobContext* job_ctx) {
    // CRITICAL FIX: Wait for shuffle to finish with proper synchronization
    {
        std::unique_lock<std::mutex> lock(job_ctx->shuffled_vectors_mutex);
        job_ctx->shuffle_cv.wait(lock, [job_ctx] { 
            return job_ctx->shuffle_finished.load(); 
        });
    }
    
    // Now do the reduce work
    while (true) {
        // Get next vector index atomically
        uint64_t vector_index = job_ctx->next_vector_index.fetch_add(1);
        
        // Check if we're done
        if (vector_index >= job_ctx->shuffled_vectors.size()) {
            break;
        }
        
        // Process this vector
        const std::vector<std::pair<K2*, V2*>>& current_vector = job_ctx->shuffled_vectors[vector_index];
        
        // Call reduce function
        job_ctx->client.reduce(&current_vector, job_ctx);
        
        // Update progress - use incremented index
        job_ctx->updateProgress(vector_index + 1);
    }
}

//=========================================

void threadFunction(JobContext* job_ctx, int thread_id) {
    ThreadContext* thread_ctx = job_ctx->thread_contexts[thread_id];
    
    // Map phase
    mapPhase(job_ctx, thread_ctx);
    
    // Sort phase
    sortPhase(thread_ctx);
    
    // CRITICAL FIX: Add barrier after sort to ensure all threads finish sorting
    job_ctx->barrier->barrier();
    
    // Only thread 0 does shuffle
    if (thread_id == 0) {
        shufflePhase(job_ctx);
    }
    
    // CRITICAL FIX: Add another barrier to ensure shuffle is complete before reduce
    job_ctx->barrier->barrier();
    
    // All threads participate in reduce
    reducePhase(job_ctx);
}

//============ EX3 FUNCTIONS ===============

void emit2(K2* key, V2* value, void* context) {
    ThreadContext* thread_ctx = static_cast<ThreadContext*>(context);
    // Each thread has its own vector - no synchronization needed as specified
    thread_ctx->intermediate_vec.emplace_back(key, value);
    
    // Update intermediate counter atomically
    thread_ctx->job_ctx->intermediate_counter.fetch_add(1);
}

void emit3(K3* key, V3* value, void* context) {
    JobContext* job_ctx = static_cast<JobContext*>(context);
    std::lock_guard<std::mutex> lock(job_ctx->output_mutex);
    job_ctx->output_vec.emplace_back(key, value);
}

JobHandle startMapReduceJob(const MapReduceClient& client,
    const InputVec& inputVec, OutputVec& outputVec,
    int multiThreadLevel) {
    
    try {
        JobContext* job_ctx = new JobContext(client, inputVec, outputVec, multiThreadLevel);
        
        // Register job globally BEFORE starting threads
        JobHandle handle = static_cast<JobHandle>(job_ctx);
        {
            std::lock_guard<std::mutex> lock(global_mutex);
            job_map[handle] = job_ctx;
        }
        
        // Start threads AFTER registration
        for (int i = 0; i < multiThreadLevel; ++i) {
            job_ctx->threads.emplace_back(threadFunction, job_ctx, i);
        }
        
        return handle;
    } catch (const std::exception& e) {
        std::cout << "system error: " << e.what() << std::endl;
        exit(1);
    }
}

void waitForJob(JobHandle job) {
    JobContext* job_ctx = nullptr;

    // Get job context without holding global lock during join
    {
        std::lock_guard<std::mutex> global_lock(global_mutex);
        auto it = job_map.find(job);
        if (it == job_map.end()) {
            return;
        }
        job_ctx = it->second;
    }

    // Prevent double joining using atomic boolean
    // Join threads outside of global lock to prevent deadlock
    bool expected = false;
    if (job_ctx->threads_joined.compare_exchange_strong(expected, true)) {
        for (auto& thread : job_ctx->threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
}

void getJobState(JobHandle job, JobState* state) {
    if (!state) {
        return;
    }
    
    JobContext* job_ctx = nullptr;
    
    // Minimize lock scope - copy job context pointer
    {
        std::lock_guard<std::mutex> lock(global_mutex);
        auto it = job_map.find(job);
        if (it == job_map.end()) {
            state->stage = UNDEFINED_STAGE;
            state->percentage = 0.0f;
            return;
        }
        job_ctx = it->second;
    }
    
    // Call getState outside of global lock
    job_ctx->getState(state);
}

void closeJobHandle(JobHandle job) {
    // Wait for job completion FIRST, outside any locks
    waitForJob(job);

    // Minimize lock scope and delete outside lock
    JobContext* job_ctx = nullptr;
    {
        std::lock_guard<std::mutex> lock(global_mutex);
        auto it = job_map.find(job);
        if (it != job_map.end()) {
            job_ctx = it->second;
            job_map.erase(it); // Remove from map while holding lock
        }
    }
    
    // Delete outside of any locks to prevent deadlock during destructor
    if (job_ctx) {
        delete job_ctx;
    }
}