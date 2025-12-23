

# %%
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()
hf_token = os.getenv("hf_access_token")

print("="*80)
print("LOADING LORA FINE-TUNED MODEL")
print("="*80)


model_save_path = "tutorial_using_llama_3B/llama3.2-base-customer-service-chat-final"

# 1. Load base model
print("\n1. Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=hf_token
)
print(f"   ✅ Base model loaded")

# 2. Load LoRA adapters
print("\n2. Loading LoRA adapters...")
model = PeftModel.from_pretrained(
    base_model,
    model_save_path,
    torch_dtype=torch.bfloat16,
)
print(f"   ✅ LoRA adapters loaded")


#%%
# 3. Setup tokenizer
print("\n3. Setting up tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_save_path, token=hf_token)
tokenizer.padding_side = 'left' #Note that for generation we should pad left

tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)
print(f"   ✅ Tokenizer configured")

# 4. Resize embeddings. 
print("\n4. Resizing embeddings...")
model.resize_token_embeddings(len(tokenizer))
print(f"   ✅ Resized to {len(tokenizer)}")

print(f'\nlen tokenizer: {len(tokenizer)}  embedding size: {model.get_input_embeddings().weight.shape[0]}')

# 5. Verify token IDs
print("\n5. Verifying token IDs...")
eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id

print(f"   <|eot_id|>: {eot_id}")
print(f"   eos_token_id: {eos_id}")
print(f"   pad_token_id: {pad_id}")

print("\n✅ MODEL READY!")
print("="*80)

#%%
# 6. Build bad words list
print("\nBuilding bad words list...")
bad_words_ids = []
for i in range(256):
    token_name = f"<|reserved_special_token_{i}|>"
    token_id = tokenizer.convert_tokens_to_ids(token_name)
    if token_id != tokenizer.unk_token_id:
        bad_words_ids.append([token_id])
print(f"✅ Blocking {len(bad_words_ids)} reserved tokens")

# 7. Test generation with FIXED parameters
print("\n" + "="*80)
print("TESTING GENERATION (FIXED)")
print("="*80)

SYSTEM_MESSAGE = "You are a helpful and professional customer service assistant. Provide clear, accurate, and friendly responses to customer inquiries."

test_queries = [
    "How do I track my order?",
    "I want to return my product",
    "What's your refund policy?",
]

for i, query in enumerate(test_queries, 1):
    print(f"\n[TEST {i}/{len(test_queries)}]")
    print("="*80)
    
    test_messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": query}
    ]
    
    test_prompt = tokenizer.apply_chat_template(
        test_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    print(f"Query: {query}")
    print("Generating...")
    
    # FIXED GENERATION PARAMETERS
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=pad_id,
        eos_token_id=[eot_id, eos_id],  # ← Multiple EOS tokens
        bad_words_ids=bad_words_ids,
        repetition_penalty=1.1,  # ← Prevent repetition
    )
    
    # Decode and extract response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract assistant's response more carefully
    if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
        assistant_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        
        # Stop at first occurrence of end token
        for end_token in ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]:
            if end_token in assistant_response:
                assistant_response = assistant_response.split(end_token)[0]
                break
        
        assistant_response = assistant_response.strip()
    else:
        assistant_response = full_response
    
    print(f"\nResponse:\n{assistant_response}")
    
    # Validation
    issues = []
    if "<|reserved_special_token" in assistant_response:
        issues.append("Reserved tokens in response")
    if any(ord(char) > 127 and char not in "áéíóúñ¿¡" for char in assistant_response):
        issues.append("Unexpected unicode characters")
    if len(assistant_response.split("<|start_header_id|>")) > 1:
        issues.append("Multiple conversation turns")
    
    if issues:
        print("⚠️  ISSUES:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Clean response")
    
    print("="*80)

print("\n✅ ALL TESTS COMPLETE!")


# %%
# Merge LoRA weights into base model for standalone deployment
merged_model = model.merge_and_unload()

# Save merged model
merged_output = "llama3-customer-service-merged"
merged_model.save_pretrained(merged_output)
tokenizer.save_pretrained(merged_output)

# Now you can load without PeftModel:
# model = AutoModelForCausalLM.from_pretrained(merged_output)

# %%
