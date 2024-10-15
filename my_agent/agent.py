from langgraph.graph import StateGraph, END, START, MessagesState
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def get_weather(city: str):
    """Get the weather for a specific city"""
    return f"It's sunny in {city}!"


raw_model = ChatOpenAI()
model = raw_model.with_structured_output(get_weather)


class SubGraphState(MessagesState):
    city: str


def model_node(state: SubGraphState):
    result = model.invoke(state["messages"])
    return {"city": result["city"]}


def weather_node(state: SubGraphState):
    result = get_weather.invoke({"city": state["city"]})
    return {"messages": [{"role": "assistant", "content": result}]}


subgraph = StateGraph(SubGraphState)
subgraph.add_node(model_node)
subgraph.add_node(weather_node)
subgraph.add_edge(START, "model_node")
subgraph.add_edge("model_node", "weather_node")
subgraph.add_edge("weather_node", END)
subgraph = subgraph.compile(interrupt_before=["weather_node"])

# Define the config
from typing import Literal
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver


memory = MemorySaver()


class RouterState(MessagesState):
    route: Literal["weather", "other"]


class Router(TypedDict):
    route: Literal["weather", "other"]


router_model = raw_model.with_structured_output(Router)


def router_node(state: RouterState):
    system_message = "Classify the incoming query as either about weather or not."
    messages = [{"role": "system", "content": system_message}] + state["messages"]
    route = router_model.invoke(messages)
    return {"route": route["route"]}


def normal_llm_node(state: RouterState):
    response = raw_model.invoke(state["messages"])
    return {"messages": [response]}


def route_after_prediction(
    state: RouterState,
) -> Literal["weather_graph", "normal_llm_node"]:
    if state["route"] == "weather":
        return "weather_graph"
    else:
        return "normal_llm_node"


graph = StateGraph(RouterState)
graph.add_node(router_node)
graph.add_node(normal_llm_node)
graph.add_node("weather_graph", subgraph)
graph.add_edge(START, "router_node")
graph.add_conditional_edges("router_node", route_after_prediction)
graph.add_edge("normal_llm_node", END)
graph.add_edge("weather_graph", END)
graph = graph.compile()



class GrandfatherState(MessagesState):
    to_continue: bool


def router_node(state: GrandfatherState):
    # Dummy logic that will always continue
    return {"to_continue": True}


def route_after_prediction(state: GrandfatherState):
    if state["to_continue"]:
        return "graph"
    else:
        return END


grandparent_graph = StateGraph(GrandfatherState)
grandparent_graph.add_node(router_node)
grandparent_graph.add_node("graph", graph)
grandparent_graph.add_edge(START, "router_node")
grandparent_graph.add_conditional_edges(
    "router_node", route_after_prediction, ["graph", END]
)
grandparent_graph.add_edge("graph", END)
grandparent_graph = grandparent_graph.compile()