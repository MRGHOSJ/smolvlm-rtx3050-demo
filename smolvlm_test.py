import cv2
from PIL import Image
import torch
import time
import gc
from transformers import AutoProcessor, AutoModelForImageTextToText

# ========== GPU OPTIMIZED FOR RTX 3050 6GB ==========
MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
MAX_TOKENS = 50
IMAGE_SIZE = 384

print("=" * 60)
print("🚀 SmolVLM GPU Optimized Demo for RTX 3050")
print("=" * 60)

# Check and display GPU info
if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU Detected: {gpu_name}")
    print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
    print(f"🎯 Using device: {device}")
    
    # Clear cache
    torch.cuda.empty_cache()
else:
    print("❌ CUDA not available - using CPU (check PyTorch installation)")
    device = torch.device("cpu")
    exit()

# Load processor with correct image size
print("\n📥 Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    size={"longest_edge": IMAGE_SIZE}  # Set image size
)

# Load model
print("📥 Loading model with GPU optimizations...")

try:
    # Use dtype instead of torch_dtype (fixes warning)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,  # Fixed: use dtype instead of torch_dtype
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("✅ Model loaded in float16 precision")
    
except Exception as e:
    print(f"⚠️ Float16 failed: {e}")
    print("Trying with float32...")
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("✅ Model loaded in float32 precision")

# Ensure model is on GPU
model = model.to(device)
print(f"📊 Model loaded on: {model.device}")

# Print memory usage
allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
print(f"📊 GPU Memory used: {allocated:.2f} GB / {gpu_memory:.1f} GB")
print(f"📊 GPU Memory reserved: {reserved:.2f} GB")

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found")
    exit()

print("\n🎮 Controls:")
print("   SPACE = Capture & Analyze")
print("   'q'   = Quit")
print("\n⚡ First capture will be slower (warm-up)")

# Memory monitoring function
def print_gpu_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"📊 GPU Memory: {allocated:.2f} GB allocated, {cached:.2f} GB cached")

# Image processing function - FIXED VERSION
def process_image(frame):
    """Process image on GPU"""
    try:
        # Convert and resize
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Resize maintaining aspect ratio
        pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # FIXED: Use the correct chat template format
        # For SmolVLM, we need to use the specific format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in one sentence."}
                ]
            }
        ]
        
        # Apply chat template
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Process the image with text
        # The key is to pass images as a list and text separately
        inputs = processor(
            images=[pil_image],  # Images as list
            text=prompt,         # Text prompt
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Debug: Check what we're sending
        print(f"📊 Input keys: {list(inputs.keys())}")
        if 'pixel_values' in inputs:
            print(f"📊 Pixel values shape: {inputs['pixel_values'].shape}")
        if 'input_ids' in inputs:
            print(f"📊 Input IDs shape: {inputs['input_ids'].shape}")
        
        # Move to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Mixed precision for speed
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    temperature=None,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True  # Enable KV cache for speed
                )
        
        # Decode
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract just the assistant's response
        response = result.replace(prompt, "").strip()
        
        # Clean up
        del inputs, generated_ids
        
        return response
        
    except Exception as e:
        print(f"❌ Processing error: {str(e)[:200]}")
        return f"Error: {str(e)[:50]}"

# Alternative simpler approach
def process_image_simple(frame):
    """Simpler approach that often works better"""
    try:
        # Convert image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # VERY SIMPLE approach - use the processor's chat handling
        # Create a conversation with image
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "What's in this image?"}
                ]
            },
        ]
        
        # Let processor handle everything
        inputs = processor(conversation, return_tensors="pt").to(device)
        
        # Generate
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
        )
        
        # Decode
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract response after the last assistant tag
        response = result.split("Assistant:")[-1].strip()
        
        return response
        
    except Exception as e:
        return f"Error: {str(e)[:50]}"

# Even simpler direct approach
def process_image_direct(frame):
    """Direct image-to-text without complex chat templates"""
    try:
        # Convert image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Direct prompt (no chat template)
        prompt = "Describe this image:"
        
        # Process
        inputs = processor(
            text=prompt,
            images=[pil_image],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Generate
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
        )
        
        # Decode
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Remove the prompt
        response = result.replace(prompt, "").strip()
        
        return response
        
    except Exception as e:
        return f"Error: {str(e)[:50]}"

# Main loop
processing = False
last_process_time = 0
cooldown = 1  # seconds between captures
method = 3  # Try different methods: 1, 2, or 3

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Display webcam feed
    display = cv2.resize(frame, (640, 480))
    
    # Add overlay text
    cv2.putText(display, f"{gpu_name} - SmolVLM Demo", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 255), 2)
    
    cv2.putText(display, "SPACE=capture | q=quit", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 0), 2)
    
    cv2.putText(display, f"Method: {method} (press 1/2/3 to change)", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (200, 200, 0), 1)
    
    current_time = time.time()
    if current_time - last_process_time < cooldown:
        remaining = cooldown - (current_time - last_process_time)
        cv2.putText(display, f"Wait: {remaining:.1f}s", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 165, 255), 2)
    
    cv2.imshow('SmolVLM - RTX 3050', display)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Change method
    if key == ord('1'):
        method = 1
        print(f"\n📊 Changed to method 1")
    elif key == ord('2'):
        method = 2
        print(f"\n📊 Changed to method 2")
    elif key == ord('3'):
        method = 3
        print(f"\n📊 Changed to method 3")
    
    if key == ord(' ') and not processing:
        current_time = time.time()
        if current_time - last_process_time < cooldown:
            continue
            
        processing = True
        last_process_time = current_time
        
        print("\n" + "=" * 40)
        print(f"📸 Capturing image (Method {method})...")
        
        # Show processing indicator
        cv2.putText(display, "Processing on GPU...", 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 255), 2)
        cv2.imshow('SmolVLM - RTX 3050', display)
        cv2.waitKey(1)
        
        start_time = time.time()
        
        try:
            # Process on GPU with selected method
            if method == 1:
                response = process_image(frame)
            elif method == 2:
                response = process_image_simple(frame)
            else:  # method 3
                response = process_image_direct(frame)
                
            process_time = time.time() - start_time
            
            print(f"✅ Processed in {process_time:.3f} seconds")
            print(f"🤖: {response}")
            print_gpu_stats()
            
            # Display result
            result_frame = cv2.resize(frame, (640, 480))
            
            # Add response text (word wrap)
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_pos = 30
            max_width = 35
            
            words = response.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line) + len(word) + 1 <= max_width:
                    current_line += (" " if current_line else "") + word
                else:
                    lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Show up to 4 lines
            for i, line in enumerate(lines[:4]):
                # Semi-transparent background
                overlay = result_frame.copy()
                cv2.rectangle(overlay, 
                            (5, y_pos + i*25 - 20), 
                            (635, y_pos + i*25 + 5), 
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, result_frame, 0.3, 0, result_frame)
                
                # White text
                cv2.putText(result_frame, line, 
                           (10, y_pos + i*25), 
                           font, 0.5, (255, 255, 255), 1)
            
            # Add stats
            cv2.putText(result_frame, f"Time: {process_time:.3f}s | Method: {method}", 
                       (10, 460), font, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Result - Press any key', result_frame)
            cv2.waitKey(3000)
            cv2.destroyWindow('Result - Press any key')
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        processing = False
        
    elif key == ord('q'):
        break

# Final cleanup
cap.release()
cv2.destroyAllWindows()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("🎉 Demo completed successfully!")
print(f"📊 Final GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print("=" * 60)