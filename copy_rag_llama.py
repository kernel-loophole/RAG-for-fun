from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM,AutoTokenizer
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch
def generate_response(model, tokenizer, input_text, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer_nf4 = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model_nf4 = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", quantization_config=nf4_config)
text=generate_response(model_nf4, tokenizer_nf4, "helo learning")
print(text)
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_4bit=True, device_map="auto")
# generator = pipeline('text-generation', model="mistralai/Mistral-7B-Instruct-v0.3",quantization_config=nf4_config)
# generator("What are we having for dinner?")
