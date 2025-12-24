# %%
# imports
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

model_save_path = "llama3-customer-service-merged"
print("\n1. Loading merged (base + LoRA/PEFT adapters)  model...")

model = AutoModelForCausalLM.from_pretrained(
    model_save_path,
    device_map="auto",
    dtype=torch.bfloat16,
    token=hf_token
)

# %%
# Verify token IDs and tokenizer vs embedding size
tokenizer = AutoTokenizer.from_pretrained(model_save_path, token=hf_token)
print("\n5. Verifying token IDs...")
eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id

print(f"   <|eot_id|>: {eot_id}")
print(f"   eos_token_id: {eos_id}")
print(f"   pad_token_id: {pad_id}")


print(f'\nlen tokenizer: {len(tokenizer)}  embedding size: {model.get_input_embeddings().weight.shape[0]}')

print("\n✅ MODEL READY!")
print("="*80)

# %%
# 6. Build bad words list
print("\nBuilding bad words list...")
bad_words_ids = []
for i in range(256):
    token_name = f"<|reserved_special_token_{i}|>"
    token_id = tokenizer.convert_tokens_to_ids(token_name)
    if token_id != tokenizer.unk_token_id:
        bad_words_ids.append([token_id])
print(f"✅ Blocking {len(bad_words_ids)} reserved tokens")



# %%
# a resusabel fuction for generation
def generate_customer_service_response(query, model, tokenizer, bad_words_ids, system_message=None):
    """Generate a customer service response for a given query."""
    
    if system_message is None:
        system_message = "You are a helpful and professional customer service assistant. Provide clear, accurate, and friendly responses to customer inquiries."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        # eos_token_id=[
        #     tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        #     tokenizer.eos_token_id
        # ],
        bad_words_ids=bad_words_ids,
        repetition_penalty=1.1,
    )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    assistant_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    
    for end_token in ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]:
        if end_token in assistant_response:
            assistant_response = assistant_response.split(end_token)[0]
            break
    
    return assistant_response.strip()

#%%
# test generation 

SYSTEM_MESSAGE = "You are a helpful and professional customer service assistant. Provide clear, accurate, and friendly responses to customer inquiries."

test_queries = [
    "How do I track my order?",
    "I want to return my product",
    "What's your refund policy?",
    "your service is terrible, you're a bunch of crooks"
]

for query in test_queries:
    print(f"Query: {query}")
    print("Generating...")
    assistant_response = generate_customer_service_response(
        query,
        model,
        tokenizer,
        bad_words_ids
    )

    print(f"\nResponse:\n{assistant_response}")

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
# %%
