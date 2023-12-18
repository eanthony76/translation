import os
import torch
import gradio as gr
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from ea200_codes import ea_codes


def load_models():
    # build model and tokenizer
    model_name_dict = {
		  'nllb-distilled-600M': 'facebook/nllb-200-distilled-600M',
                  #'nllb-1.3B': 'facebook/nllb-200-1.3B',
                  #'nllb-distilled-1.3B': 'facebook/nllb-200-distilled-1.3B',
                  #'nllb-3.3B': 'facebook/nllb-200-3.3B',
                  }

    model_dict = {}

    for call_name, real_name in model_name_dict.items():
        print('\tLoading model: %s' % call_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(real_name)
        tokenizer = AutoTokenizer.from_pretrained(real_name)
        model_dict[call_name+'_model'] = model
        model_dict[call_name+'_tokenizer'] = tokenizer

    return model_dict


def translation(source, target, text):
    if len(model_dict) == 2:
        model_name = 'nllb-distilled-600M'

    start_time = time.time()
    source = ea_codes[source]
    target = ea_codes[target]

    model = model_dict[model_name + '_model']
    tokenizer = model_dict[model_name + '_tokenizer']

    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source, tgt_lang=target, device=0)
    output = translator(text, max_length=500)

    end_time = time.time()

    full_output = output
    output = output[0]['translation_text']
    result = {'inference_time': end_time - start_time,
              'source': source,
              'target': target,
              'result': output,
              'full_output': full_output}
    return result


if __name__ == '__main__':
    print('\tinit models')

    global model_dict

    model_dict = load_models()
    
    # define gradio demo
    lang_codes = list(ea_codes.keys())
    #inputs = [gr.inputs.Radio(['nllb-distilled-600M', 'nllb-1.3B', 'nllb-distilled-1.3B'], label='NLLB Model'),
    inputs = [gr.Dropdown(lang_codes, value='English', label='Source'),
              gr.Dropdown(lang_codes, value='English', label='Target'),
              gr.Textbox(lines=5, label="Input text"),
              ]

    outputs = gr.Textbox()

    title = "Translation Tool"

    description = f"Details: https://github.com/facebookresearch/fairseq/tree/nllb."

    gr.Interface(translation,
                 inputs,
                 outputs,
                 title=title,
                 description=description,
                 examples_per_page=50,
                 ).launch(server_name='0.0.0.0', share=True)


