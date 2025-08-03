## What’s the purpose of this code?
This code builds a system where agents can send messages to each other inside the same Python process — like a post office for your app, but all happening in memory (no external server).

It’s useful when different parts of your app (like document parsers, storage handlers, chat agents, etc.) need to talk to each other in a structured and organized way.

## Key Pieces
1. `MCPMessage`
    > Imagine this as a letter you send from one department to another.

    - It includes info like:
        - Who sent the message
        - Who should receive it
        - What the message is about
        - A unique ID
        - Optional metadata

2. `MCPTransport`
    > Think of this as a blueprint (interface) for how messages should be passed around.

    - It has 3 required methods:
        1. `send_message()` → to send a message.
        2. `receive_messages()` → to read messages meant for you.
        3. `subscribe()` → to register a listener (like: "ping me whenever I get a message").

    - It doesn't do anything on its own — it just defines the structure.

3. `InMemoryTransport`
    > This is a working version of the "blueprint" above, using memory instead of a network.

    - **What it does:**  
        - It keeps one queue per agent type. 
        - When you send a message, it puts that message in the receiver's queue. 
        - If someone has "subscribed" to that agent type, it also notifies them. 
        - It stores all messages in a list called message_history. 
        - It allows cleaning up old messages (messages older than 24 hours).

    - Example:
        - You call send_message() from Agent A to Agent B.
        - It puts the message into Agent B’s inbox (queue).
        - If B has a “listener”, it gets notified immediately.

4. `MCPMessageBus`
    > Think of this as the central manager — it uses the transport to actually coordinate the messages.

    It wraps around the transport layer and gives some helpful tools:

    - Key Methods:
        - `start()` → turn on the message bus.
        - `stop()` → turn it off.
        - `send_message()` → send a single message.
        - `broadcast_message()` → send the same message to many agents.
        - `request_response()` → send a message, and wait for a specific reply.

5. `Global Message Bus`
    - At the bottom of the code, we have:

        ```python
        _message_bus: Optional[MCPMessageBus] = None

        async def get_message_bus() -> MCPMessageBus:
        ```
    - This ensures that only one message bus exists and is shared across your system — a global inbox manager, so to speak.