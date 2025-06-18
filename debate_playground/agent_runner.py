from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os
import gc

# Force CPU usage if GPU memory is insufficient
device = "cpu"  # Start with CPU to avoid memory issues

model_name = "Qwen/Qwen2-0.5B"  # Updated model name
cache_dir = os.path.expanduser("~/.cache/huggingface/qwen_debate")

print("🔄 Loading model and tokenizer from cache...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    cache_dir=cache_dir,
    trust_remote_code=True
)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Check CUDA availability first
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    try:
        # Try GPU first with device_map
        print("🔄 Attempting GPU loading with device_map...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"  # Let accelerate handle device placement
        )
        device = "cuda"  # Device is managed by accelerate
        print("✅ Model loaded on GPU with accelerate")
        
    except Exception as e:
        print(f"⚠️ GPU loading failed: {e}")
        print("🔄 Falling back to CPU...")
        # Fallback to CPU without device_map
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True
        )
        device = "cpu"
        print("✅ Model loaded on CPU")

else:
    # CPU only
    print("🔄 Loading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for CPU
        low_cpu_mem_usage=True
    )
    device = "cpu"
    print("✅ Model loaded on CPU")

model.eval()
print(f"✅ Model ready on {device}")

def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def generate_response(system_prompt, topic, history, max_length=500):
    """Generate response with memory management"""
    try:
        # Build prompt
        prompt = f"{system_prompt}\n\nTopic: {topic}\n"
        for turn in history:
            # Use 'literature_response' and 'science_response' keys
            lit_resp = turn.get('literature_response', turn.get('lit', '[No literature response]'))
            sci_resp = turn.get('science_response', turn.get('sci', '[No science response]'))
            prompt += f"\nTurn {turn['turn']}\nLiterature: {lit_resp}\nScience: {sci_resp}\n"
        
        # Encourage detailed responses
        prompt += f"\nProvide a detailed response (up to {max_length} words):\n"
        
        # Tokenize with length limits
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=1024,  # Limit input length
            truncation=True
        )
        
        # Move inputs to the same device as the model
        if cuda_available:
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Generate with conservative settings
        with torch.inference_mode(): 
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new part
        response_part = response[len(prompt):].strip()
        
        # Cleanup
        del inputs, outputs
        cleanup_memory()
        
        return response_part if response_part else "No response generated."
        
    except Exception as e:
        print(f"❌ Error in generation: {e}")
        cleanup_memory()
        return f"Error generating response: {str(e)}"

# System prompts
literature_prompt = "You are a philosopher. Respond poetically and metaphorically in a detailed manner.but be precise and limit your answers within 5 to 7 sentences."
science_prompt = "You are a scientist. Respond with logical, evidence-based points in a detailed manner.but be precise and limit your answers within 5 to 7 sentences."
judge_prompt = "You are a debate judge. Score both responses 1-10. Format: Literature: X/10, Science: Y/10. Reason: [brief explanation on why you awarded that score to them]"

def lit_agent(topic, history):
    """Literature agent with error handling"""
    try:
        return generate_response(literature_prompt, topic, history)
    except Exception as e:
        print(f"❌ Literature agent error: {e}")
        return f"Literature perspective on {topic}: [Error in generation]"

def sci_agent(topic, history):
    """Science agent with error handling"""
    try:
        return generate_response(science_prompt, topic, history)
    except Exception as e:
        print(f"❌ Science agent error: {e}")
        return f"Scientific perspective on {topic}: [Error in generation]"

def judge_agent(topic, history):
    """Judge agent with robust parsing"""
    try:
        judgment = generate_response(judge_prompt, topic, history)
        
        # More robust score extraction
        lit_match = re.search(r"Literature:?\s*(\d+)", judgment, re.IGNORECASE)
        sci_match = re.search(r"Science:?\s*(\d+)", judgment, re.IGNORECASE)
        
        lit_score = int(lit_match.group(1)) if lit_match else 5
        sci_score = int(sci_match.group(1)) if sci_match else 5
        
        # Ensure scores are within valid range
        lit_score = max(1, min(10, lit_score))
        sci_score = max(1, min(10, sci_score))
        
        return {
            "lit_score": lit_score,
            "sci_score": sci_score,
            "reason": judgment
        }
        
    except Exception as e:
        print(f"❌ Judge agent error: {e}")
        return {
            "lit_score": 5,
            "sci_score": 5,
            "reason": f"Error in judgment: {str(e)}"
        }

# Memory monitoring function
def get_memory_info():
    """Get current memory usage"""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_reserved = torch.cuda.memory_reserved(0)
            return f"GPU: {gpu_allocated/1024**3:.1f}GB/{gpu_memory/1024**3:.1f}GB allocated, {gpu_reserved/1024**3:.1f}GB reserved"
        except:
            return "GPU available but can't access memory info"
    return "Using CPU"

# Print initial memory status
print(f"💾 Memory status: {get_memory_info()}")