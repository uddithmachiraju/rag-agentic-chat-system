import uuid 
import pytest 
import asyncio
from datetime import datetime, UTC, timedelta

from src.mcp.protocol import MCPMessage, AgentType, MessageType, MessageStatus
from src.mcp.transport import InMemoryTransport, MCPMessageBus

@pytest.fixture
def sample_message():
    return MCPMessage(
        message_id = str(uuid.uuid4()),
        sender = AgentType.INGESTION,
        receiver = AgentType.INGESTION,
        message_type = MessageType.CONTEXT_REQUEST,
        status = MessageStatus.PENDING, 
        payload = {"input": "Sample input"},
        metadata = {},
        trace_id = str(uuid.uuid4()) 
    )

import pytest
from pprint import pprint  # For clean print of dicts/lists

@pytest.mark.asyncio
async def test_send_and_receive_message(sample_message):

    transport = InMemoryTransport()
    await transport.send_message(sample_message)
    messages = await transport.receive_messages(sample_message.receiver)

    assert len(messages) == 1, f"Expected 1 message, got {len(messages)}"
    assert messages[0].payload == sample_message.payload, (
        f"Expected payload {sample_message.payload}, got {messages[0].payload}"
    )

@pytest.mark.asyncio
async def test_subscribe_callback_called(sample_message):
    transport = InMemoryTransport()

    callback_called = []

    def callback(msg):
        callback_called.append(msg)

    await transport.subscribe(AgentType.INGESTION, callback)
    await transport.send_message(sample_message)
    await asyncio.sleep(0.1)

    assert len(callback_called) == 1, f"Expected 1 callback, got {len(callback_called)}"
    assert callback_called[0].message_id == sample_message.message_id, (
        f"Expected message ID: {sample_message.message_id}, got: {callback_called[0].message_id}"
    )

@pytest.mark.asyncio 
async def test_message_history_filtering(sample_message):
    transport = InMemoryTransport()
    await transport.send_message(sample_message)

    history = await transport.get_message_history(trace_id = sample_message.trace_id)
    assert len(history) == 1
    assert history[0].trace_id == sample_message.trace_id 

    histiry_by_agent = await transport.get_message_history(agent_type = AgentType.INGESTION)
    assert any(msg.sender == AgentType.INGESTION for msg in histiry_by_agent) 

@pytest.mark.asyncio
async def test_message_bus_send_and_receive(sample_message):
    bus = MCPMessageBus(InMemoryTransport())
    await bus.start() 

    await bus.send_message(sample_message)
    messages = await bus.transport.receive_messages(AgentType.INGESTION)
    assert len(messages) == 1
    assert messages[0].payload["input"] == "Sample input"

@pytest.mark.asyncio 
async def test_broadcast_message(sample_message):
    bus = MCPMessageBus(InMemoryTransport())
    await bus.start()

    targets = [AgentType.INGESTION]
    result = await bus.broadcast_message(sample_message, targets)
    assert all(result.values()) 

    for target in targets:
        messages = await bus.transport.receive_messages(target)
        assert len(messages) == 1
        assert messages[0].receiver == target 

@pytest.mark.asyncio
async def test_request_response():
    bus = MCPMessageBus(InMemoryTransport())
    await bus.start()

    request = MCPMessage(
        message_id = str(uuid.uuid4()),
        sender = AgentType.INGESTION,
        receiver = AgentType.INGESTION,
        message_type = MessageType.CONTEXT_REQUEST,
        status = MessageStatus.PENDING, 
        payload = {"input": "Ping"},
        metadata = {},
        trace_id = str(uuid.uuid4()) 
    )

    async def simulate_response():
        await asyncio.sleep(0.2)
        response = MCPMessage(
            sender = AgentType.INGESTION,
            receiver = AgentType.INGESTION,
            message_type = MessageType.CONTEXT_RESPONSE,
            status = MessageStatus.PENDING, 
            payload = {"output": "Pong"},
            metadata = {"original_message_id": request.message_id},
            trace_id = request.trace_id
        )
        await bus.transport.send_message(response)

    asyncio.create_task(simulate_response())

    resp = await bus.request_response(request, timeout=2.0)

    assert resp is not None, "Did not receive any response"
    assert resp.payload["output"] == "Pong", "Response payload did not contain expected 'pong'"