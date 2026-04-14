import os
import yaml
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    # Backward compatibility for older LangChain versions.
    from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from jinja2 import Template

def init_llm(model_name=None):
    # Define model paths
    model_paths = {
        'Meta-Llama-3.1-8B-Instruct-Q4_K_M': 'model/Meta-Llama-3.1-8B-Instruct-Q4_K_M/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
        'Meta-Llama-3.1-8B-Instruct-Q6_K': 'model/Meta-Llama-3.1-8B-Instruct-Q6_K/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf',
        'komodo-7b-base.Q5_0': 'model/komodo-7b-base.Q5_0/komodo-7b-base.Q5_0.gguf'
    }
    
    # Use default model if none specified
    if not model_name or model_name not in model_paths:
        model_name = 'Meta-Llama-3.1-8B-Instruct-Q4_K_M'
    
    model_path = model_paths[model_name]
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found at {model_path}")
    
    # Performance mode optimized for RTX 4060
    return LlamaCpp(
        model_path=model_path,
        n_gpu_layers=40,           # Aggressive GPU utilization
        n_ctx=2048,                # Reduced from 8192 for speed
        n_batch=256,               # Reduced from default 512
        temperature=0.3,           # Balanced for format compliance
        top_p=0.9,                 # Standard sampling
        top_k=40,                  # Standard top-k
        n_threads=8,
        verbose=False,             # Disable verbose for cleaner output
        repeat_penalty=1.1,        # Standard repeat penalty
    )

def load_prompts(file_path='prompts_chat.yaml'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_data = yaml.safe_load(f)
        print("Loaded prompts raw data:", loaded_data)
        prompts = loaded_data.get('prompts', {})
        print("Prompts before validation:", prompts)
        
        if not prompts:
            raise ValueError("No prompts loaded from prompts.yaml")
            
        return prompts
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing {file_path}: {e}")
        raise

def render_chat_prompt(messages, template_str, add_generation_prompt=False):
    template = Template(template_str)
    return template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt
    )

class ChatPromptAdapter:
    def __init__(self, llm, prompt_template):
        self.llm = llm
        self.template = prompt_template
    
    def invoke(self, messages):
        prompt = render_chat_prompt(
            messages=messages,
            template_str=self.template,
            add_generation_prompt=True
        )
        return self.llm.invoke(prompt)

def get_chat_chain(llm, prompt_id="chat_friendly"):
    prompts = load_prompts()
    prompt_config = prompts.get(prompt_id)
    
    if not prompt_config:
        available_prompts = list(prompts.keys())
        raise ValueError(f"Prompt ID '{prompt_id}' not found. Available prompts: {available_prompts}")
    
    if prompt_config["type"] == "chat":
        return ChatPromptAdapter(llm, prompt_config["template_jinja"])
    else:
        raise ValueError(f"Unsupported prompt type for chat: {prompt_config['type']}")

# Chat session management
class ChatSession:
    def __init__(self, model_name=None):
        self.llm = init_llm(model_name)
        self.chain = get_chat_chain(self.llm)
        self.history = []
    
    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
    
    def get_response(self, user_input):
        self.add_message("user", user_input)
        
        try:
            print("Processing user input:", user_input)
            response = self.chain.invoke(self.history)
            print("Raw model response:", response)
            
            # Clean up the response if needed
            if isinstance(response, str):
                response = response.strip()
                # Remove any unwanted escape characters and formatting
                response = response.replace('\\n', '\n')
                response = response.split('User:')[0].strip()  # Only take the assistant's response
                response = response.replace('Assistant:', '').strip()
            
            self.add_message("assistant", response)
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return error_msg
