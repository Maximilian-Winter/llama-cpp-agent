from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent import StructuredOutputAgent
import llama_cpp

llama = llama_cpp.Llama.from_pretrained(
    repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
    filename="*Q4_K_M.gguf",
    verbose=False,
    n_gpu_layers = -1,
    n_ctx = 3060
)

def to_llama_cpp_prompt(Character1_introduction, Character2_introduction, interactive_content, Character1, Character2, round = 8):
    Character_introduction = Character1_introduction + Character2_introduction
    input_ = 'Based on the above background information, select an appropriate dialogue scene and generate a dialogue between {} and {} to express the "{}" part.'.format(Character1, Character2, interactive_content)
    return "{}\n{}\nIt is required that the participants in the dialogue are only {} and {}. The two people dialogue alternately. Only the two of them are mentioned in the dialogue. When using pronouns, Use only you and I as mutual terms of address, at least {} rounds of dialogue are generated. ".format(
        Character_introduction, input_, Character1, Character2, round)

def generate_chat_context(Character1_introduction, Character2_introduction,
                            interactive_content, Character1, Character2,
                            structured_output_agent):
    prompt = to_llama_cpp_prompt(Character1_introduction, Character2_introduction, interactive_content, Character1, Character2)

    class User(Enum):
        character1 = Character1
        character2 = Character2

    class ChatItem(BaseModel):
        '''
        words spoken by a person
        '''
        name: User = Field(...,
            description="person"
        )
        sentence:str =  Field(...,
            description="words"
        )

    class Conversation_List(BaseModel):
        """
        Represents a conversation between two people
        """
        Background: str = Field(...,
            description="Explain the time and place of the conversation, relevant character settings, and the cause of the conversation."
        )
        ChatContext: List[ChatItem] = Field(...,
            description="dialogue between characters"
        )

    out = structured_output_agent.create_object(Conversation_List, prompt)
    return out.json()

Character1_introduction = '''
Amy: A college student who is very passionate about e-sports games, often participate in competitions and have multiple competition trophies.
'''

Character2_introduction = '''
Ben: He is a game company executive responsible for developing and publishing e-sports games and has in-depth knowledge of e-sports events.
'''

Character1 = "Amy"
Character2 = "Ben"

interactive_content = '''
Amy admires Ben's technology very much and expresses her appreciation for his entrepreneurial success.
'''

from llama_cpp_agent.providers.llama_cpp_python import LlmProviderId, LlamaCppPythonProvider
provider = LlamaCppPythonProvider(llama)

structured_output_agent = StructuredOutputAgent(
    provider, debug_output=True,
    messages_formatter_type = MessagesFormatterType.MISTRAL
)

out_json = generate_chat_context(Character1_introduction, Character2_introduction,
                            interactive_content, Character1, Character2,
                            structured_output_agent)
