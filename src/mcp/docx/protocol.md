## What is the purpose of this code?
This code defines a message-passing system for our RAG pipeline.
Agents (like Ingestion, Retrieval, LLM) use this system to send structured messages to each other, such as:
- Asking for document processing
- Sending back results
- Reporting errors 

Think of it like a messaging protocol or internal post office for your agents to talk.

## Key pieces
1. `MCPMessage (model)`
    > A standard message object that every agent uses to communicate.

    - Contains:
        - `message_id`: Unique ID for this message 
        - `trace_id`: Used to track message flow across services 

        - `sender & receiver`: Who sent it and who should receive it

        - `message_type`: What kind of message (REQUEST, RESPONSE, etc.)

        - `status`: Processing state (PENDING, SUCCESS, ERROR, etc.)

        - `payload`: Main data/content

        - `metadata`: Extra info

2. `create_mcp_message`
    > Create a new message from one agent to another.
    - Used when you want to initiate communication — e.g., ask another agent to do something.

3. `create_response_message`
    > Reply to a message — like answering a question.
    - Used when you finished processing and want to send back results.

4. `create_error_message`
    > Send an error message when something goes wrong.
    - Used when an agent fails to process a request and wants to notify the sender.
    - What it does:
        - Marks the message as ERROR
        - Includes error details (message + optional code)
        - Keeps the original trace

5. `DocumentProcessingRequest`
    > Defines what data is required to ask for document processing.
    - Includes:
        - `file_path`, `file_name`, `file_type`: Info about the file
        - `user_id`: Optional – who sent the file
        - `processing_options`: Instructions
    - Used as:
        - Payload inside an MCPMessage.

6. `DocumentProcessingResponse`
    > Defines the structure of the response after processing a document.
    - Includes:
        - `document_id`
        - `chunks_created`: Number of text pieces created
        - `metadata`
        - `processing_time`
        - `success`: True/False
        - `error_message`: Optional if failed