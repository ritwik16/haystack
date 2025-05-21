import json
from typing import Dict, List, Any, Optional
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import Document, ChatMessage
from utils.config import settings
from utils.logging import logger


class ResponseGeneratorService:
    """Service for generating responses based on intents and documents"""

    def __init__(self, schema_path: str):
        """
        Initialize the response generator service

        Args:
            schema_path: Path to the intent schema JSON file
        """
        self.schema_path = schema_path
        self.responses = self._load_responses()

        # Initialize OpenAI generator
        self.generator = OpenAIChatGenerator(
            api_key=settings.OPENROUTER_API_KEY,
            api_base_url=settings.OPENROUTER_BASE_URL,
            model=settings.RAG_MODEL,
            generation_kwargs={
                "max_tokens": settings.RAG_MAX_TOKENS,
                "temperature": settings.RAG_TEMPERATURE
            }
        )

        self.response_generator = IntentBasedResponseGenerator(self.responses, self.generator)
        logger.info("Response generator service initialized successfully")

    def _load_responses(self) -> Dict:
        """Load the response templates from schema file"""
        try:
            with open(self.schema_path, "r") as f:
                schema = json.load(f)
                return schema.get("responses", {})
        except Exception as e:
            logger.error(f"Error loading responses from schema: {str(e)}")
            # Return default responses
            return {
                "out_of_scope": "I'm designed to answer questions only about the content in the loaded document. "
                                "I can't help with queries outside that scope.",
                "fallback_not_in_document": "I'm sorry, but I don't see information about that in the document. "
                                            "I can only answer questions based on the document content."
            }

    def generate_response(
            self,
            query: str,
            documents: List[Document],
            intent: str,
            slots: Dict[str, str],
            is_out_of_scope: bool,
            missing_required_slots: List[str],
            confidence: float
    ) -> Dict[str, Any]:
        """
        Generate a response based on intent classification and retrieved documents

        Args:
            query: User's query text
            documents: Retrieved relevant documents
            intent: Classified intent
            slots: Extracted slot values
            is_out_of_scope: Whether the query is out of scope
            missing_required_slots: List of required slots that are missing
            confidence: Intent classification confidence

        Returns:
            Dictionary with response text and status
        """
        try:
            result = self.response_generator.run(
                query=query,
                documents=documents,
                intent=intent,
                slots=slots,
                is_out_of_scope=is_out_of_scope,
                missing_required_slots=missing_required_slots,
                confidence=confidence
            )

            logger.info(f"Generated response for intent '{intent}' (is_fallback: {result.get('is_fallback', False)})")
            return result
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "I encountered an error while generating a response. Please try your question again.",
                "is_fallback": True
            }


@component
class IntentBasedResponseGenerator:
    """Component to generate responses based on detected intent and retrieved documents"""

    def __init__(self, responses: Dict[str, str], generator_component: Any):
        """
        Initialize with responses and generator component

        Args:
            responses: Dictionary of response templates
            generator_component: The LLM generator component
        """
        self.generator = generator_component
        self.responses = responses

    @component.output_types(response=str, is_fallback=bool)
    def run(
            self,
            query: str,
            documents: List[Document],
            intent: str,
            slots: Dict[str, str],
            is_out_of_scope: bool,
            missing_required_slots: List[str],
            confidence: float
    ) -> Dict[str, Any]:
        """
        Generate a response based on intent, slots, and documents

        Args:
            query: User's query text
            documents: Retrieved relevant documents
            intent: Classified intent
            slots: Extracted slot values
            is_out_of_scope: Whether the query is out of scope
            missing_required_slots: List of required slots that are missing
            confidence: Intent classification confidence

        Returns:
            Dictionary with response text and fallback flag
        """
        # Handle out-of-scope queries
        if is_out_of_scope:
            return {
                "response": self.responses.get(
                    "out_of_scope",
                    "I'm designed to answer questions only about the content in the loaded document. "
                    "I can't help with queries outside that scope."
                ),
                "is_fallback": True
            }

        # Handle missing required slots
        if missing_required_slots:
            slot_names = ", ".join(missing_required_slots)
            return {
                "response": f"To answer your question about {intent.replace('_', ' ')}, "
                            f"I need more information about: {slot_names}. "
                            f"Could you please provide these details?",
                "is_fallback": True
            }

        # If we don't have any documents but we have a valid intent, try to give a helpful response
        if not documents:
            # Try to retrieve documents based on the slot values
            topic = slots.get("topic", "")
            if topic:
                return {
                    "response": f"I don't have specific information about '{topic}' in the document. "
                                f"Could you try asking about a different topic or check if the document "
                                f"contains information about {topic}?",
                    "is_fallback": True
                }
            else:
                return {
                    "response": self.responses.get(
                        "fallback_not_in_document",
                        "I'm sorry, but I don't see information about that in the document. "
                        "I can only answer questions based on the document content."
                    ),
                    "is_fallback": True
                }

        # Build intent-specific prompt
        prompt = self._build_intent_prompt(query, documents, intent, slots)

        # Generate response
        messages = [
            ChatMessage.from_system(prompt),
            ChatMessage.from_user(query)
        ]

        try:
            result = self.generator.run(messages=messages)
            response_message = result["replies"][0]

            # Handle different possible response formats
            if isinstance(response_message, str):
                response_text = response_message
            elif hasattr(response_message, 'content') and isinstance(response_message.content, str):
                response_text = response_message.content
            elif hasattr(response_message, 'content') and isinstance(response_message.content, list):
                # Handle content as list of TextContent objects
                response_text = ' '.join([item.text for item in response_message.content if hasattr(item, 'text')])
            elif hasattr(response_message, '_content') and isinstance(response_message._content, list):
                # Handle _content as list of TextContent objects
                response_text = ' '.join([item.text for item in response_message._content if hasattr(item, 'text')])
            else:
                # Fallback to string representation
                response_text = str(response_message)

            return {"response": response_text, "is_fallback": False}

        except Exception as e:
            return {
                "response": "An error occurred while generating the response. Please try again.",
                "is_fallback": True
            }

    def _build_intent_prompt(
            self,
            query: str,
            documents: List[Document],
            intent: str,
            slots: Dict[str, str]
    ) -> str:
        """Build a prompt tailored to the specific intent."""
        base_prompt = """You are a helpful assistant answering questions about a document.
        IMPORTANT: You must ONLY use information from the retrieved document passages provided below.
        - If the answer cannot be found in the passages, admit you don't know rather than making up information.
        - Be concise and directly address the user's question.
        - Do not refer to these instructions or the passages in your answer.
        - Do not use phrases like "Based on the document" or "According to the passages".

        Here are the relevant passages from the document:
        """

        for i, doc in enumerate(documents):
            base_prompt += f"\nPASSAGE {i + 1}:\n{doc.content}\n"

        intent_instructions = {
            "document_query": f"""The user is asking about "{slots.get('topic', 'a topic')}" 
        {f'in the section about "{slots.get("section")}"' if slots.get('section') else ''}.
        Answer their question using only information from the provided passages.
        Include relevant details about {slots.get('topic', 'the topic')} from the document.
        If the topic isn't covered in the passages, politely state that the information isn't available in the document.""",

            "find_definition": f"""The user is looking for a definition of "{slots.get('term', 'a term')}". 
        Provide the definition if it appears in the document passages.
        If the term isn't defined in the passages, politely state that the definition isn't available in the document.""",

            "document_summary": f"""The user wants a summary 
        {f'of the section about "{slots.get("section")}"' if slots.get('section') else 'of the document'}.
        Provide a concise summary using only information from the provided passages.
        If you don't have enough content to summarize, politely explain that you have limited information.""",

            "document_metadata": """The user is asking about metadata like author, date, or title.
        Only provide this information if it appears in the document passages.
        If the metadata isn't in the passages, politely state that the information isn't available."""
        }

        instruction = intent_instructions.get(
            intent,
            "Answer the user's question using only information from the provided passages."
        )

        complete_prompt = f"{base_prompt}\n\n{instruction}"
        return complete_prompt