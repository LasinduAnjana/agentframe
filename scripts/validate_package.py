#!/usr/bin/env python3
"""
Package validation script for AgentFrame.

This script validates that the package is correctly structured and
all components can be imported successfully.
"""

import sys
import os
import importlib.util
from pathlib import Path

def validate_package_structure():
    """Validate that all required files and directories exist."""
    print("🔍 Validating package structure...")

    required_files = [
        "pyproject.toml",
        "setup.py",
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "src/agentframe/__init__.py",
        "src/agentframe/core/__init__.py",
        "src/agentframe/models/__init__.py",
        "src/agentframe/tools/__init__.py",
        "src/agentframe/memory/__init__.py",
        "src/agentframe/utils/__init__.py",
        "tests/conftest.py",
        "examples/basic_agent.py",
        "docs/getting_started.md",
        "docs/publishing.md"
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    print("✅ All required files present")
    return True

def validate_imports():
    """Validate that core components can be imported."""
    print("\n🔍 Validating imports...")

    # Add src to path for testing
    src_path = Path("src").absolute()
    sys.path.insert(0, str(src_path))

    try:
        # Test core imports
        print("  Testing core imports...")
        import agentframe
        from agentframe import Agent, AgentConfig
        from agentframe import ModelConfig, OpenAIModel, GeminiModel, ClaudeModel
        from agentframe import tool, BaseTool, ToolRegistry
        from agentframe import ChatHistory, MessageType
        from agentframe import Intent, IntentParser

        print("✅ Core imports successful")

        # Test version
        print(f"  AgentFrame version: {agentframe.__version__}")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def validate_tool_decorator():
    """Validate that the @tool decorator works correctly."""
    print("\n🔍 Validating @tool decorator...")

    try:
        from agentframe import tool

        @tool
        def test_tool(input_text: str) -> str:
            """A test tool for validation."""
            return f"Processed: {input_text}"

        # Check if tool has the correct attributes
        if not hasattr(test_tool, '_agentframe_tool'):
            print("❌ Tool decorator didn't attach _agentframe_tool attribute")
            return False

        tool_instance = test_tool._agentframe_tool
        if tool_instance.name != "test_tool":
            print(f"❌ Tool name mismatch: expected 'test_tool', got '{tool_instance.name}'")
            return False

        # Test tool execution
        result = tool_instance.execute({"input_text": "hello"})
        if not result.success:
            print(f"❌ Tool execution failed: {result.error}")
            return False

        print("✅ Tool decorator working correctly")
        return True

    except Exception as e:
        print(f"❌ Tool decorator validation failed: {e}")
        return False

def validate_model_configs():
    """Validate that model configurations work correctly."""
    print("\n🔍 Validating model configurations...")

    try:
        from agentframe import ModelConfig, ValidationError
        from agentframe.utils import validate_configuration

        # Test valid configuration
        config = ModelConfig(
            api_key="test-key-12345",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )

        config_dict = config.to_dict()
        if "api_key" not in config_dict:
            print("❌ ModelConfig.to_dict() missing api_key")
            return False

        print("✅ Model configuration working correctly")
        return True

    except Exception as e:
        print(f"❌ Model configuration validation failed: {e}")
        return False

def validate_docker_setup():
    """Validate Docker configuration files."""
    print("\n🔍 Validating Docker setup...")

    required_docker_files = [
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore"
    ]

    missing = []
    for file in required_docker_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print(f"❌ Missing Docker files: {missing}")
        return False

    print("✅ Docker configuration files present")
    return True

def main():
    """Run all validation checks."""
    print("🚀 AgentFrame Package Validation")
    print("=" * 50)

    checks = [
        validate_package_structure,
        validate_imports,
        validate_tool_decorator,
        validate_model_configs,
        validate_docker_setup
    ]

    passed = 0
    failed = 0

    for check in checks:
        try:
            if check():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Check {check.__name__} crashed: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 Validation Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All validations passed! Package is ready for distribution.")
        return 0
    else:
        print("⚠️  Some validations failed. Please fix issues before publishing.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)