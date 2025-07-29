"""Base class for all query types."""

from __future__ import annotations

from typing import Any

from metasim.sim.base import BaseSimHandler
from metasim.sim.genesis import GenesisHandler
from metasim.sim.isaacgym import IsaacgymHandler
from metasim.sim.isaaclab import IsaaclabHandler
from metasim.sim.mjx import MJXHandler
from metasim.sim.pybullet import PybulletHandler
from metasim.sim.pyrep import PyrepHandler
from metasim.sim.sapien import Sapien2Handler, Sapien3Handler


class BaseQueryType() :
    suported_handlers = []

    def __init__(self, **kwargs) :

        self.handler = None
        self.query_options = kwargs

    def bind_handler(self, handler: BaseSimHandler, *args: Any, **kwargs) :
        """
        By default, all queries will be binded to the handler at handler.launch() stage.
        By default, the queries will be given the handler instance as the first argument.
        For different simulation handlers, the queries may be binded at different stages.
        For different simulation handlers, the queries may be given other optional arguments.
        You can also pass locals() to bind_handler() to access local variables in the handler.
        """
        if handler.__class__ not in self.suported_handlers:
            raise ValueError(f"Handler {handler.__class__.__name__} not supported for query {self.__class__.__name__}")
        self.handler = handler

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        By default, the query has no arguments.
        The query will be called in handler.get_extra() stage.
        The query should return a specific value.
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
