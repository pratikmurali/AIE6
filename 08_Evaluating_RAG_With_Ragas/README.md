
<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 8: Evaluating RAG with Ragas</h1>

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 8: Pre-Work](https://www.notion.so/Session-8-RAG-Evaluation-and-Assessment-1c8cd547af3d81d08f7cf5521d0253bb?pvs=4#1c8cd547af3d816583d6c23183b6f87f) | [Session 8: RAG Evaluation and Assessment](https://www.notion.so/Session-8-RAG-Evaluation-and-Assessment-1c8cd547af3d81d08f7cf5521d0253bb) | Coming soon! | [Session 8 Slides](https://www.canva.com/design/DAGjadKGqcw/0Gff9K2EwbOb3lX14un3uw/edit?utm_content=DAGjadKGqcw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 8: RAG Evaluation and Assessment](https://forms.gle/ujAQLqx2ZHMWTUH79) | [AIE6 Feedback 4/24](https://forms.gle/wA7p89e6svCgjtr58) |

In today's assignment, we'll be creating Synthetic Data, and using it to benchmark (and improve) a LCEL RAG Chain.

#### ❓🙋🏽‍♂️ Question and Answers

#### ❓ Question: 
Describe in your own words what a "trace" is.
#### 📍 Answer
<b>Logging</b> records specific events or messages within a single service. <b>Tracing</b>, on the other hand, tracks the journey of a request or transaction as it flows through multiple services in an agentic system, offering end-to-end visibility and helping to identify performance bottleneck. The journey of every request is called a trace.

#### ❓ Question: 

Describe *how* each of the above metrics are calculated. This will require you to read the documentation for each metric.

##### 👨🏽‍💻 Tool Call Accuracy Metric

<b>How it's calculated:</b>
The metric examines each tool call in a conversation trace
It compares the name of the tool called and the arguments passed to expected/reference tool calls
By default, it uses exact string matching for comparing tool names and arguments
The metric returns a score between 0 and 1:
<ul>
   <li> 1: Perfect match - the model called the correct tools with the correct arguments in the correct order </li>
   <li> 0: No match - the model called different tools or used incorrect arguments </li>
</ul>
The tool call accuracy can be customized to use similarity metrics instead of exact matching. For example, the documentation mentions you can use <b>NonLLMStringSimilarity()</b> as the arg_comparison_metric to better handle natural language strings in arguments.
<pre>
<code language="python">
metric = ToolCallAccuracy()
metric.arg_comparison_metric = NonLLMStringSimilarity()
</code>
</pre>
This metric is particularly important for agent systems as it directly measures if the model is effectively invoking the tools it has access to.

##### 👨🏽‍💻 Agent Goal Accuracy Metric

<b> How it's calculated: </b>
The metric can be calculated in two ways:
<ul>
<li> <b>1. With reference (AgentGoalAccuracyWithReference):</b>
<ul>
<li>Requires user_input (conversation trace) and reference (ideal outcome)</li>
<li>An LLM evaluator compares the final state of the conversation with the reference</li>
<li>The evaluator determines if the agent achieved the goal specified in the reference
Returns 1 if the goal was achieved, 0 if not </li>
</ul>
</li>
<li> <b>2. Without reference (AgentGoalAccuracyWithoutReference): </b>
<ul>
<li>Only requires user_input (conversation trace)</li>
<li>The desired outcome is inferred from the human interactions in the workflow</li>
<li>An LLM evaluator analyzes the conversation to determine if the apparent goal was achieved
Returns 1 if the goal was achieved, 0 if not</li>
</ul>
</ul>
In both cases, the evaluation is performed by an LLM which acts as a judge to determine if the agent successfully accomplished what the user asked for.


##### 👨🏽‍💻 Topic Adherence Metric

<b> How it's calculated: </b>
The metric requires:
<ul>
<li>user_input (conversation trace)</li>
<li>reference_topics (list of topics the agent should adhere to)</li>
</ul>
It calculates three scores:
<ul>
<li><b>Precision:</b> Proportion of answered queries that adhere to reference topics </li>
<pre>
    Precision = (Queries answered that adhere to reference topics) / (Total queries answered)
</pre> </li>
<li> <b>Recall:</b> Proportion of on-topic queries that were correctly answered (not refused)
<pre>
   Recall = (Queries answered that adhere to reference topics) / (Queries answered that adhere to reference topics + Queries refused that should have been answered)
</pre>
</li>
<li>
<b>F1 Score:</b> Harmonic mean of precision and recall
<pre>
 F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
</pre>
</li>
</ul>
The default metric reported depends on the mode parameter, which can be set to "precision", "recall", or "f1".
The scoring is done by an LLM that examines the conversation and determines whether each exchange adheres to the specified reference topics.

When evaluated on our example in the notebook where an eagle question was asked of a metals-focused agent, the Topic Adherence score was 0 because the conversation went entirely off-topic into bird flight speeds rather than staying on the topic of metals.


#### ❓ Question: 

Which system performed better, on what metrics, and why?

#### 📍 Answer

After Cohere reranking.The answer relevancy and noise sensitivity metrics improved.

##### RAGA metrics Without Cohere Rerank

<span style="background-color: #90EE90">'context_recall': 0.7514,</span> 
<span style="background-color: #90EE90">'faithfulness': 0.7772, </span>
<span style="background-color: #90EE90">'factual_correctness': 0.5833,</span>
<span style="background-color: orange"> 'answer_relevancy': 0.8026,</span> 
<span style="background-color: #90EE90"> 'context_entity_recall': 0.5160,</span>
<span style="background-color: orange">'noise_sensitivity_relevant': 0.2841</span>


##### RAGA metrics With Cohere Rerank

<span style="background-color: orange"> 'context_recall': 0.6503, </span>
<span style="background-color: orange">'faithfulness': 0.6810, </span>
<span style="background-color: orange">'factual_correctness': 0.5342, </span>
<span style="background-color: #90EE90">'answer_relevancy': 0.8717, </span>
<span style="background-color: orange">'context_entity_recall': 0.4048,</span>
<span style="background-color: #90EE90">'noise_sensitivity_relevant': 0.2516</span>



- 🤝 Breakout Room #1
  1. Task 1: Installing Required Libraries
  2. Task 2: Set Environment Variables
  3. Task 3: Synthetic Dataset Generation for Evaluation using Ragas
  4. Task 4: Evaluating our Pipeline with Ragas
  5. Task 6: Making Adjustments and Re-Evaluating

  The notebook Colab link is located [here](https://colab.research.google.com/drive/1-t4POIFJI-SWF1lmoBOPETZZqgWCTV4Y?usp=sharing)

- 🤝 Breakout Room #2
  1. Task 1: Building a ReAct Agent with Metal Price Tool
  2. Task 2: Implementing the Agent Graph Structure
  3. Task 3: Converting Agent Messages to Ragas Format
  4. Task 4: Evaluating Agent Performance using Ragas Metrics
     - Tool Call Accuracy
     - Agent Goal Accuracy  
     - Topic Adherence

The notebook Colab link is located [here](https://colab.research.google.com/drive/1KQm7nA_zTaCyjaAeAacjqanMPv03um7T?usp=sharing)

## Ship 🚢

The completed notebook!

<details>
<summary>🚧 BONUS CHALLENGE 🚧 (OPTIONAL)</summary>

> NOTE: Completing this challenge will provide full marks on the assignment, regardless of the completion of the notebook. You do not need to complete this in the notebook for full marks.

##### **MINIMUM REQUIREMENTS**:

1. Baseline `LangGraph RAG` Application using `NAIVE RETRIEVAL`
2. Baseline Evaluation using `RAGAS METRICS`
  - [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html)
  - [Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/answer_relevance.html)
  - [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/context_precision.html)
  - [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/context_recall.html)
  - [Answer Correctness](https://docs.ragas.io/en/stable/concepts/metrics/answer_correctness.html)
3. Implement a `SEMANTIC CHUNKING STRATEGY`.
4. Create an `LangGraph RAG` Application using `SEMANTIC CHUNKING` with `NAIVE RETRIEVAL`.
5. Compare and contrast results.

##### **SEMANTIC CHUNKING REQUIREMENTS**:

Chunk semantically similar (based on designed threshold) sentences, and then paragraphs, greedily, up to a maximum chunk size. Minimum chunk size is a single sentence.

Have fun!
</details>

### Deliverables

- A short Loom of the notebook, and a 1min. walkthrough of the application in full

## Share 🚀

Make a social media post about your final application!

### Deliverables

- Make a post on any social media platform about what you built!

Here's a template to get you started:

```
🚀 Exciting News! 🚀

I am thrilled to announce that I have just built and shipped Synthetic Data Generation, benchmarking, and iteration with RAGAS & LangChain! 🎉🤖

🔍 Three Key Takeaways:
1️⃣ 
2️⃣ 
3️⃣ 

Let's continue pushing the boundaries of what's possible in the world of AI and question-answering. Here's to many more innovations! 🚀
Shout out to @AIMakerspace !

#LangChain #QuestionAnswering #RetrievalAugmented #Innovation #AI #TechMilestone

Feel free to reach out if you're curious or would like to collaborate on similar projects! 🤝🔥
```
