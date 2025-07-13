"""
Base agent class for the AI Research Agent system
Provides common functionality and error handling for all agents
"""

import asyncio
import time
import traceback
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from config.settings import Settings
from utils.logger import setup_logger, log_execution_time, LoggingContext


class AgentStatus(Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class AgentResult:
    """Standardized agent result structure"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    agent_name: Optional[str] = None


class BaseAgent(ABC):
    """
    Base class for all agents in the research system
    Provides common functionality, error handling, and monitoring
    """
    
    def __init__(self, settings: Settings, name: str = "BaseAgent"):
        self.settings = settings
        self.name = name
        self.logger = setup_logger(f"{__name__}.{name}")
        self.status = AgentStatus.IDLE
        self.session_data = {}
        self.metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0
        }
        
        # Initialize agent-specific configurations
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize agent-specific configurations"""
        pass
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Process input data and return results
        This method must be implemented by all child agents
        """
        pass
    
    async def execute(self, input_data: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Main execution method with comprehensive error handling and monitoring
        """
        start_time = time.time()
        operation_id = f"{self.name}_{int(time.time())}"
        
        try:
            self.status = AgentStatus.RUNNING
            self.metrics["total_operations"] += 1
            
            with LoggingContext(self.logger, f"{self.name} execution", operation_id=operation_id):
                # Validate input
                if not await self.validate_input(input_data):
                    raise ValueError(f"Invalid input data for {self.name}")
                
                # Log execution start
                await self.log_activity("execution_start", {
                    "operation_id": operation_id,
                    "input_size": len(str(input_data)),
                    "kwargs": kwargs
                })
                
                # Execute the main processing
                result = await self.process(input_data)
                
                # Update metrics
                execution_time = time.time() - start_time
                self.metrics["successful_operations"] += 1
                self.metrics["total_execution_time"] += execution_time
                self.metrics["average_execution_time"] = (
                    self.metrics["total_execution_time"] / self.metrics["total_operations"]
                )
                
                # Ensure result is properly formatted
                if not isinstance(result, AgentResult):
                    result = AgentResult(
                        success=True,
                        data=result if isinstance(result, dict) else {"result": result},
                        execution_time=execution_time,
                        agent_name=self.name
                    )
                else:
                    result.execution_time = execution_time
                    result.agent_name = self.name
                
                self.status = AgentStatus.COMPLETED
                
                # Log successful completion
                await self.log_activity("execution_success", {
                    "operation_id": operation_id,
                    "execution_time": execution_time,
                    "result_size": len(str(result.data)) if result.data else 0
                })
                
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.status = AgentStatus.ERROR
            self.metrics["failed_operations"] += 1
            self.metrics["total_execution_time"] += execution_time
            
            # Create error result
            error_result = await self.handle_error(e, "execution", {
                "operation_id": operation_id,
                "execution_time": execution_time,
                "input_data": str(input_data)[:500]  # Limit size for logging
            })
            
            return AgentResult(
                success=False,
                error=error_result["error"],
                metadata=error_result.get("metadata", {}),
                execution_time=execution_time,
                agent_name=self.name
            )
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data structure and content
        Override in child classes for specific validation
        """
        try:
            if not isinstance(input_data, dict):
                self.logger.warning(f"Input data is not a dictionary: {type(input_data)}")
                return False
            
            # Check for required fields (override in child classes)
            required_fields = self.get_required_fields()
            missing_fields = [field for field in required_fields if field not in input_data]
            
            if missing_fields:
                self.logger.warning(f"Missing required fields: {missing_fields}")
                return False
            
            # Additional validation can be added in child classes
            return await self.custom_validation(input_data)
            
        except Exception as e:
            self.logger.error(f"Input validation error: {str(e)}")
            return False
    
    def get_required_fields(self) -> List[str]:
        """
        Get list of required fields for input validation
        Override in child classes
        """
        return []
    
    async def custom_validation(self, input_data: Dict[str, Any]) -> bool:
        """
        Custom validation logic for specific agents
        Override in child classes
        """
        return True
    
    async def log_activity(self, activity: str, details: Dict[str, Any] = None):
        """Log agent activity with structured data"""
        activity_data = {
            "agent": self.name,
            "activity": activity,
            "status": self.status.value,
            "details": details or {},
            "timestamp": time.time()
        }
        
        self.logger.log_structured("info", f"{self.name}: {activity}", activity_data)
    
    async def handle_error(self, error: Exception, context: str = "", 
                          additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle errors gracefully with comprehensive logging"""
        error_data = {
            "agent": self.name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "additional_data": additional_data or {},
            "status": self.status.value
        }
        
        self.logger.log_error_with_context(error, f"{self.name} - {context}", error_data)
        
        return {
            "success": False,
            "error": f"{self.name} error in {context}: {str(error)}",
            "agent": self.name,
            "context": context,
            "error_type": type(error).__name__,
            "metadata": error_data
        }
    
    async def retry_operation(self, operation_func, max_retries: int = 3, 
                             delay: float = 1.0, backoff_factor: float = 2.0):
        """
        Retry operation with exponential backoff
        """
        for attempt in range(max_retries):
            try:
                result = await operation_func()
                if attempt > 0:
                    await self.log_activity("retry_success", {
                        "attempt": attempt + 1,
                        "max_retries": max_retries
                    })
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:
                    await self.log_activity("retry_failed", {
                        "final_attempt": attempt + 1,
                        "max_retries": max_retries,
                        "error": str(e)
                    })
                    raise
                
                wait_time = delay * (backoff_factor ** attempt)
                await self.log_activity("retry_attempt", {
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "wait_time": wait_time,
                    "error": str(e)
                })
                
                await asyncio.sleep(wait_time)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information"""
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "status": self.status.value,
            "metrics": self.metrics.copy(),
            "settings": {
                "search": self.settings.get_search_config(),
                "verification": self.settings.get_verification_config(),
                "quality": self.settings.get_quality_config()
            },
            "session_data_size": len(self.session_data),
            "uptime": time.time() - getattr(self, '_start_time', time.time())
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "total_operations": self.metrics["total_operations"],
            "successful_operations": self.metrics["successful_operations"],
            "failed_operations": self.metrics["failed_operations"],
            "success_rate": (
                self.metrics["successful_operations"] / max(self.metrics["total_operations"], 1)
            ) * 100,
            "average_execution_time": self.metrics["average_execution_time"],
            "total_execution_time": self.metrics["total_execution_time"]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the agent"""
        health_status = {
            "agent_name": self.name,
            "status": self.status.value,
            "healthy": True,
            "issues": [],
            "metrics": self.get_performance_metrics(),
            "timestamp": time.time()
        }
        
        # Check for common issues
        if self.metrics["failed_operations"] > self.metrics["successful_operations"]:
            health_status["healthy"] = False
            health_status["issues"].append("High failure rate")
        
        if self.metrics["average_execution_time"] > 60.0:  # 1 minute threshold
            health_status["healthy"] = False
            health_status["issues"].append("Slow execution times")
        
        if self.status == AgentStatus.ERROR:
            health_status["healthy"] = False
            health_status["issues"].append("Agent in error state")
        
        return health_status
    
    async def reset_agent(self):
        """Reset agent state and metrics"""
        self.status = AgentStatus.IDLE
        self.session_data.clear()
        self.metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0
        }
        
        await self.log_activity("agent_reset", {"timestamp": time.time()})
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', status='{self.status.value}')"
    
    def __str__(self) -> str:
        return f"{self.name} Agent (Status: {self.status.value})"


class AgentManager:
    """Manager class for coordinating multiple agents"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = setup_logger("AgentManager")
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_queue = asyncio.Queue()
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the manager"""
        self.agents[agent.name] = agent
        self.logger.info(f"Registered agent: {agent.name}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self.agents.get(name)
    
    async def execute_agent(self, agent_name: str, input_data: Dict[str, Any]) -> AgentResult:
        """Execute specific agent"""
        agent = self.get_agent(agent_name)
        if not agent:
            return AgentResult(
                success=False,
                error=f"Agent '{agent_name}' not found",
                agent_name=agent_name
            )
        
        return await agent.execute(input_data)
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all agents"""
        health_results = {}
        overall_healthy = True
        
        for name, agent in self.agents.items():
            health_result = await agent.health_check()
            health_results[name] = health_result
            if not health_result["healthy"]:
                overall_healthy = False
        
        return {
            "overall_healthy": overall_healthy,
            "agents": health_results,
            "total_agents": len(self.agents),
            "healthy_agents": sum(1 for r in health_results.values() if r["healthy"]),
            "timestamp": time.time()
        }
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        total_operations = sum(agent.metrics["total_operations"] for agent in self.agents.values())
        total_successes = sum(agent.metrics["successful_operations"] for agent in self.agents.values())
        total_failures = sum(agent.metrics["failed_operations"] for agent in self.agents.values())
        
        return {
            "total_agents": len(self.agents),
            "total_operations": total_operations,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "overall_success_rate": (total_successes / max(total_operations, 1)) * 100,
            "agent_details": {
                name: agent.get_performance_metrics() 
                for name, agent in self.agents.items()
            }
        }