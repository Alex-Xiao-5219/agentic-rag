from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.nodes.responder import generate_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.state import AgentState

workflow = StateGraph(AgentState)

workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)

workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)

checkpointer = MemorySaver()
rag_agent = workflow.compile(checkpointer=checkpointer)

