from uam import logger
from uam.agents.continuous import _CAGENTS


class AgentNotFoundError(Exception):
    pass

def validate_agent(agent: str) -> None:
    """Validates an agent_name string passed
    into it. 
    Returns an AgentNotFoundError 
    if the passed agent does not match
    any available agent, else None.
    """
    
    if agent not in _CAGENTS:
        logger.error(f"`{agent}` did not match any of the "
                     "available agents.\nAvailable Agents:\n"
                     f"{_CAGENTS}"
        )
        raise AgentNotFoundError

def is_continuous(name: str) -> bool:
    return True if name in _CAGENTS else False