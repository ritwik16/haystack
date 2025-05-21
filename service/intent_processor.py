import json
import re
from typing import Dict, List, Any, Optional
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from utils.config import settings
from utils.logging import logger


class IntentProcessorService:

    def __init__(self, schema_path: str):

        self.schema_path = schema_path
        self.schema = self._load_schema()
        self.schema_prompt = self._format_schema_for_prompt()

        # Initialize OpenAI generator
        self.generator = OpenAIChatGenerator(
            api_key=settings.OPENROUTER_API_KEY,
            api_base_url=settings.OPENROUTER_BASE_URL,
            model=settings.INTENT_MODEL,
            generation_kwargs={
                "max_tokens": settings.INTENT_MAX_TOKENS,
                "temperature": settings.INTENT_TEMPERATURE
            }
        )

        self.processor = IntentSlotProcessor(self.schema_prompt, self.generator)
        logger.info("Intent processor service initialized successfully")

    def _load_schema(self) -> Dict:
        """Load the intent schema from file"""
        try:
            with open(self.schema_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading intent schema: {str(e)}")
            # Return a minimal default schema
            return {
                "intents": [
                    {
                        "name": "document_query",
                        "description": "Questions about document content",
                        "slots": [{"name": "topic", "is_required": True, "description": "The topic to query about"}],
                        "examples": ["What does the document say about X?"]
                    }
                ]
            }

    def _format_schema_for_prompt(self) -> str:
        """Format the schema for inclusion in the prompt"""
        prompt = "AVAILABLE INTENTS:\n\n"
        for intent in self.schema["intents"]:
            prompt += f"Intent: {intent['name']}\n"
            prompt += f"Description: {intent['description']}\n"
            if intent.get("slots"):
                prompt += "Slots:\n"
                for slot in intent["slots"]:
                    required = "required" if slot.get("is_required") else "optional"
                    prompt += f"  - {slot['name']} ({required}): {slot['description']}\n"
            prompt += "Examples:\n"
            for example in intent.get("examples", []):
                prompt += f"  - {example}\n"
            prompt += "\n"
        return prompt

    def process_intent(self, query: str) -> Dict[str, Any]:
        """
        Process the user's intent and extract slots

        Args:
            query: The user's query text

        Returns:
            Dictionary with intent processing results
        """
        try:
            result = self.processor.run(query=query)
            logger.info(f"Processed intent: {result['intent']} with confidence {result['confidence']}")
            return result
        except Exception as e:
            logger.error(f"Error processing intent for query '{query}': {str(e)}")
            # Return a default fallback
            return {
                "intent": "document_query",
                "slots": {"topic": self._extract_fallback_topic(query)},
                "is_out_of_scope": False,
                "missing_required_slots": [],
                "confidence": 0.6
            }

    def _extract_fallback_topic(self, query: str) -> str:
        """Extract a fallback topic from query when intent processing fails"""
        # Simple approach: use the first noun in the query
        words = query.lower().split()
        for word in words:
            if word not in ["what", "does", "the", "document", "say", "about", "tell", "me"]:
                return word
        return "bananas"  # Default fallback


@component
class IntentSlotProcessor:
    """Haystack component for processing intents and extracting slots"""

    def __init__(self, schema_prompt: str, generator_component: Any):
        """
        Initialize with schema and generator component

        Args:
            schema_prompt: Formatted schema prompt
            generator_component: The LLM generator component
        """
        self.generator = generator_component
        self.schema_prompt = schema_prompt

    @component.output_types(
        intent=str,
        slots=Dict[str, str],
        is_out_of_scope=bool,
        missing_required_slots=List[str],
        confidence=float
    )
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run intent processing on a query

        Args:
            query: The user's query text

        Returns:
            Intent processing results
        """
        # Enhanced system prompt with clearer instructions
        system_prompt = f"""You are an NLU system that identifies user intent and extracts slots from queries about documents.
IMPORTANT: ALWAYS assume queries are about document content unless they are clearly about something else.
If a query mentions "the document" or asks about topics, information, content, etc., it's ALWAYS a document_query intent.

Follow these instructions:
1. Analyze the query and match it to the most appropriate intent from the schema.
2. For document_query intent, extract 'topic' from the query - this is what the user wants to know about.
3. If the query is about finding a definition in the document, use find_definition intent.
4. If the query is about summarizing the document, use document_summary intent.
5. If the query is about document metadata like author or title, use document_metadata intent.
6. ONLY use out_of_scope for queries that are clearly NOT about document content (like weather, jokes, etc.).
7. Provide a reasonable confidence level (0.0-1.0) based on how well the query matches an intent.

{self.schema_prompt}

Query: {query}

Respond in JSON format:
{{
  "intent": "intent_name",
  "slots": {{"slot_name": "extracted_value"}},
  "is_out_of_scope": false,
  "confidence": 0.95
}}"""

        messages = [
            ChatMessage.from_system(system_prompt),
            ChatMessage.from_user(query)
        ]

        logger.debug(f"IntentProcessor query: {query}")

        try:
            result = self.generator.run(messages=messages)
            response_text = result["replies"][0]

            logger.debug(f"IntentProcessor response_text: {response_text}")

            try:
                # Extract JSON response using regex
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_json = json.loads(json_match.group(0))
                else:
                    response_json = json.loads(response_text)

                logger.debug(f"IntentProcessor parsed JSON: {response_json}")

                # Default to document_query for document-related queries if confidence is low
                intent = response_json.get("intent", "document_query")
                if "document" in query.lower() and intent == "out_of_scope":
                    intent = "document_query"
                    response_json["intent"] = "document_query"
                    response_json["is_out_of_scope"] = False
                    response_json["confidence"] = max(0.7, response_json.get("confidence", 0.7))

                # Extract slots
                slots = response_json.get("slots", {})

                # Default topic slot to query content if not specified for document_query
                if intent == "document_query" and "topic" not in slots:
                    # Extract topic from query or default to a general topic
                    topic = "general"  # Default topic
                    for word in query.lower().split():
                        if word not in ["what", "does", "the", "document", "say", "about", "tell", "me"]:
                            topic = word
                            break
                    slots["topic"] = topic

                is_out_of_scope = response_json.get("is_out_of_scope", False)
                confidence = float(response_json.get("confidence", 0.7))  # Default to reasonable confidence

                # Identify missing required slots
                missing_required_slots = []

                return {
                    "intent": intent,
                    "slots": slots,
                    "is_out_of_scope": is_out_of_scope,
                    "missing_required_slots": missing_required_slots,
                    "confidence": confidence
                }

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}, response_text: {response_text}")
                # Fallback to document_query for safety
                return {
                    "intent": "document_query",
                    "slots": {"topic": "general"},
                    "is_out_of_scope": False,
                    "missing_required_slots": [],
                    "confidence": 0.6
                }

        except Exception as e:
            logger.error(f"Generator error in IntentSlotProcessor: {str(e)}")
            # Fallback to document_query for safety
            return {
                "intent": "document_query",
                "slots": {"topic": "general"},
                "is_out_of_scope": False,
                "missing_required_slots": [],
                "confidence": 0.6
            }


# Service instance
intent_service = IntentProcessorService(settings.SCHEMA_PATH)