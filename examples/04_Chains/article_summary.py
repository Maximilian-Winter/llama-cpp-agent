# Example: Article Summary and Social Media Post
from llama_cpp_agent import AgentChainElement, AgentChain
from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppServerProvider

provider = LlamaCppServerProvider("http://localhost:8080")

agent = LlamaCppAgent(
    provider,
    debug_output=True,
    system_prompt="",
    predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL
)

article_summary = AgentChainElement(
    output_identifier="out_0",
    system_prompt="You are an article summarization assistant",
    prompt="Summarize the key points of the following article in 3-4 sentences:\n--\n{article_text}"
)


social_media_post = AgentChainElement(
    output_identifier="out_1",
    system_prompt="You are a social media manager",
    prompt="Create an engaging social media post based on the following article summary. Include relevant hashtags:\n--\n{out_0}"
)

chain = [article_summary, social_media_post]
agent_chain = AgentChain(agent, chain)
agent_chain.run_chain(additional_fields={"article_text": """### 1. Quantum Computing: The Next Frontier in Computational Power

**Introduction**
Quantum computing represents a revolutionary approach to information processing, leveraging the principles of quantum mechanics to solve problems that are intractable for classical computers. This article explores the fundamental concepts of quantum computing, its potential applications, and the challenges it faces.

**Quantum Mechanics and Computing**
Quantum computers use quantum bits, or qubits, which can exist in multiple states simultaneously, thanks to superposition. This capability, combined with entanglement—where the state of one qubit can depend on the state of another, no matter the distance between them—allows quantum computers to process a vast number of possibilities concurrently.

**Quantum Algorithms**
Several algorithms have been developed for quantum computers that show significant speed-ups over their classical counterparts. Shor’s Algorithm, for instance, can factorize large integers exponentially faster than the best-known classical algorithms, which has profound implications for cryptography. Grover's Algorithm offers a quadratic speedup for unstructured search problems.

**Applications**
Quantum computing has potential applications across various fields:
- **Cryptography**: Secure communication through quantum key distribution.
- **Drug Discovery**: Modeling molecular interactions at quantum levels to predict drug efficacy and side effects.
- **Optimization Problems**: Enhancing solutions in logistics, finance, and materials science.

**Challenges**
Despite its potential, quantum computing faces several hurdles:
- **Qubit Coherence**: Maintaining the state of qubits for sufficient time is challenging due to decoherence.
- **Error Rates**: Quantum gates are prone to errors significantly higher than conventional binary computing gates.
- **Scalability**: Building machines with enough qubits to be useful for complex problems is currently beyond our reach.

**Conclusion**
Quantum computing is still in its infancy, but it holds the promise of massive computational power. The coming decades are likely to see significant advancements in this field as researchers overcome its current limitations."""})  # Replace "..." with the actual article text
