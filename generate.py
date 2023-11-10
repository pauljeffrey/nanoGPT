from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def generate_response(model,tokenizer, prompt, do_sample=True, temperature=0.9,max_length=256):
    result = tokenizer(prompt, return_tensors="pt")
    input_ids = result.input_ids.to("cuda")
    attention_mask = result.attention_mask.to("cuda")

    gen_tokens = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=do_sample,
        temperature=temperature,
        max_length=max_length,
    )
    input_token_len = len(input_ids[0])
    gen_text = tokenizer.decode(gen_tokens[0,input_token_len:]) # you can use batch_decode also: gen_text = tokenizer.decode(gen_tokens)[0] 
    return gen_text