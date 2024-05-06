### Mapping Chain Example
This example demonstrates how to create a mapping chain to summarize 3 articles into one summary.
```python
from llama_cpp_agent.chain import AgentChainElement, MapChain
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings

model = LlamaCppEndpointSettings(completions_endpoint_url="http://127.0.0.1:8080/completion")

agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="",
    predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL
)

summary_chain = AgentChainElement("out_0", system_prompt="You are an advanced AI agent for summarizing articles", prompt="Summarize this article into bullet points:\n{item}")

combine_chain = AgentChainElement("out_1", system_prompt="You are an advanced AI agent that summarizes text", prompt="Please combine the bullet points of different summaries below, into one summary as bullet points:\n{map_output}")

map_chain = MapChain(agent, [summary_chain], [combine_chain])


article_list = [
    """### 1. Quantum Computing: The Next Frontier in Computational Power

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
Quantum computing is still in its infancy, but it holds the promise of massive computational power. The coming decades are likely to see significant advancements in this field as researchers overcome its current limitations.""",
    """### 2. Machine Learning: Transforming Data into Insights

**Introduction**
Machine Learning (ML) is a branch of artificial intelligence that focuses on building applications that can learn from data and improve their accuracy over time without being explicitly programmed. This article delves into the types of ML, key algorithms, applications, and future prospects.

**Types of Machine Learning**
- **Supervised Learning**: Models predict outputs based on input data, and learning is guided by comparing actual and predicted outputs.
- **Unsupervised Learning**: Algorithms identify patterns in data without reference to known or labeled outcomes.
- **Reinforcement Learning**: Models learn to make sequences of decisions by receiving rewards or penalties.

**Key Algorithms**
- **Linear Regression** and **Logistic Regression** for predictive modeling.
- **Decision Trees** and **Random Forests** for classification and regression tasks.
- **Neural Networks**: Used in deep learning for complex pattern recognition, such as in image and speech recognition.

**Applications**
- **Healthcare**: From diagnosing diseases to personalized medicine.
- **Finance**: For credit scoring, algorithmic trading, and risk assessment.
- **Retail**: Enhancing customer experience through personalized recommendations.

**Challenges and Future Prospects**
The field of ML is not without challenges, including data privacy concerns, the need for large labeled datasets, and the risk of creating biased models. However, ongoing research in areas like unsupervised learning, transfer learning, and the development of more robust models promises to mitigate these issues.

**Conclusion**
Machine learning continues to be a dynamic field of research and application, with the potential to impact numerous sectors profoundly.""",
    """### 3. Blockchain Technology: Decentralizing Trust

**Introduction**
Blockchain technology is best known as the backbone of cryptocurrencies like Bitcoin, but its applications extend far beyond. This article outlines the technology’s fundamentals, applications beyond finance, and the challenges it faces.

**Blockchain Basics**
A blockchain is a decentralized ledger of all transactions across a network. Each transaction is added to a "block" and linked to the previous block, forming a "chain." This structure, combined with cryptographic techniques, makes it secure and immutable.

**Key Features**
- **Decentralization**: No single point of control or failure.
- **Transparency**: Changes to the public blockchain are viewable by all parties creating transparency.
- **Immutability**: Once a transaction is recorded, it cannot be altered, increasing trust.

**Applications**
- **Supply Chain Management**: Enhancing transparency and traceability.
- **Healthcare**: Secure sharing of medical records.
- **Smart Contracts**: Automatically executing contracts when conditions are met.

**Challenges**
- **Scalability**: Current blockchain solutions, like Bitcoin, have limitations on transaction speed and volume.
- **Regulatory Issues**: Balancing the need for regulation with the ethos of decentralization.
- **Energy Consumption**: The energy requirement for "mining" transactions, particularly in networks like Bitcoin, is substantial.

**Conclusion**
Blockchain technology holds great promise for creating a more transparent and efficient world, but significant challenges must be addressed to realize its full potential."""
]


map_chain.run_map_chain(items_to_map=article_list)
```