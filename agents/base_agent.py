#!/usr/bin/env python3
"""
Base Agent
----------
Abstract base class shared by all scenario agents (EnergyScan, CalibScan, HVEqualization).

Lifecycle:
  with agent:           → load()  : load Qwen model onto MPS / CUDA / CPU
    agent.run()         → subclass-defined experiment workflow
                        → decide(context) : LLM inference → JSON parse → action
  (exit with block)     → unload(): delete model & tokenizer, flush cache

Subclasses must implement:
  _get_system_prompt()   : step-by-step workflow instructions for the LLM
  _build_state_context() : serialize current state dict into a prompt string
  run()                  : conversation loop + tool execution logic
"""

import json
import time
import torch
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MAX_CONVERSATION_HISTORY, MAX_NEW_TOKENS


class ToolFatalError(Exception):
    """Tool이 max_retries 이후에도 실패했을 때 발생."""
    pass


class BaseAgent(ABC):
    """
    모든 Agent의 기본 클래스
    
    Features:
    - State 관리
    - Conversation history 관리 (최근 N개)
    - LLM 로드/언로드 (메모리 효율)
    - Context 생성
    """
    
    def __init__(self, model_path: str, agent_name: str, io_handler=None):
        """
        Args:
            model_path: Fine-tuned 모델 경로
            agent_name: Agent 이름 (로깅용)
            io_handler: IOHandler instance (None → TerminalIO)
        """
        self.model_path = Path(model_path)
        self.agent_name = agent_name

        if io_handler is None:
            from agents.io_handler import TerminalIO
            self.io = TerminalIO()
        else:
            self.io = io_handler
        
        self.model = None
        self.tokenizer = None
        self.device = None
        self.state = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history = MAX_CONVERSATION_HISTORY
    
    # ===== Context Manager (자동 로드/언로드) =====
    
    def __enter__(self):
        """with 문 시작: 모델 로드"""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """with 문 종료: 모델 언로드"""
        self.unload()
        return False
    
    # ===== 모델 관리 =====
    
    def load(self):
        """모델 로드"""
        if self.model is not None:
            return
        
        if torch.backends.mps.is_available():
            self.device = "mps"
            print(f"  ✅ [{self.agent_name}] MPS 사용")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print(f"  ✅ [{self.agent_name}] CUDA 사용")
        else:
            self.device = "cpu"
            print(f"  ✅ [{self.agent_name}] CPU 사용")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )
        
        print(f"  ✅ [{self.agent_name}] 모델 로드 완료: {self.model_path}")
    
    def unload(self):
        """모델 언로드 (메모리 해제)"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        print(f"  🗑️  [{self.agent_name}] 모델 언로드 완료")
    
    # ===== 대화 관리 =====
    
    def add_to_history(self, role: str, content: str, metadata: Optional[Dict] = None):
        """대화 히스토리에 추가"""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            entry.update(metadata)
        
        self.conversation_history.append(entry)
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_recent_history(self, n: Optional[int] = None) -> List[Dict]:
        """최근 N개 대화 반환"""
        if n is None:
            n = self.max_history
        return self.conversation_history[-n:]
    
    # ===== Context 생성 =====
    
    @abstractmethod
    def _build_state_context(self) -> str:
        """State를 문자열로 변환 (각 Agent가 구현)"""
        pass
    
    def _build_history_context(self) -> str:
        """대화 히스토리를 문자열로 변환 (최근 대화 우선)"""
        if not self.conversation_history:
            return "(No conversation yet)"
        
        lines = []
        recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Agent"
            content = msg["content"]
            
            if role == "Agent":
                try:
                    decision = json.loads(content)
                    if "message" in decision:
                        content = decision["message"]
                    elif "tool" in decision:
                        content = f"[Tool Call: {decision['tool']}]"
                        if "update_state" in decision:
                            content += f" (Update State: {list(decision['update_state'].keys())})"
                except:
                    pass

            if content == "완료" or content.strip() == "완료":
                lines.append(f"{role}: 완료 [IMPORTANT: User confirmed completion]")
            else:
                lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def build_full_context(self, current_input: Optional[str] = None) -> str:
        """전체 context 생성"""
        parts = []
        
        parts.append("=== Current State ===")
        parts.append(self._build_state_context())
        parts.append("")
        
        parts.append("=== Recent Conversation ===")
        parts.append(self._build_history_context())
        parts.append("")
        
        if current_input:
            parts.append("=== Current User Input ===")
            parts.append(current_input)
            parts.append("")
        
        parts.append("=== Your Task ===")
        parts.append("Based on the current state and conversation, decide the next action.")
        parts.append("Output JSON with tool name and parameters.")
        
        return "\n".join(parts)
    
    # ===== LLM 호출 =====
    
    def decide(self, context: str) -> Dict[str, Any]:
        """LLM에게 다음 행동 결정을 요청하고 JSON으로 반환"""
        if self.model is None:
            raise RuntimeError(f"[{self.agent_name}] Model not loaded. Use with statement or call load().")
        
        system_prompt = self._get_system_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        try:
            json_str = generated_text.strip()
            if '{' not in json_str:
                raise json.JSONDecodeError("No JSON found", json_str, 0)
            
            # 첫 번째 '{'부터 시작하여 중괄호 매칭으로 완전한 JSON 추출
            start = json_str.index('{')
            depth = 0
            end = start
            
            for i in range(start, len(json_str)):
                char = json_str[i]
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            
            if depth != 0:
                raise json.JSONDecodeError("Incomplete JSON", json_str, end)
            
            json_str = json_str[start:end]
            
            # Python boolean → JSON boolean 변환
            json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
            
            decision = json.loads(json_str)
            return decision
        
        except (json.JSONDecodeError, ValueError) as e:
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw_output": generated_text
            }
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """System prompt 반환 (Agent별로 구현)"""
        pass
    
    # ===== Tool 재시도 =====

    def _run_tool_with_retry(self, tool_fn: Callable, tool_name: str, max_retries: int = 3) -> str:
        """tool_fn을 최대 max_retries번 자동 재시도. 모두 실패하면 에러 표시 후 사용자 '다시 시도' 대기."""
        while True:
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    return tool_fn()
                except RuntimeError as e:
                    last_error = e
                    self.log(f"[Retry {attempt}/{max_retries}] Tool '{tool_name}' 실패: {e}")
                    if attempt < max_retries:
                        time.sleep(2)
            self.io.send_tool_error(tool_name, str(last_error), max_retries)
            action = self.io.wait_for_retry()  # blocks until retry or skip
            if action == "skip":
                return f"[SKIPPED] {tool_name} 건너뜀 (사용자 요청)"

    # ===== 로깅 =====
    # stdout only — no file I/O

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{self.agent_name}] {message}", flush=True)
    
    # ===== 추상 메서드 =====
    
    @abstractmethod
    def run(self):
        """
        Agent 실행 (각 Agent가 구현)
        """
        pass
