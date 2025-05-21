import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from utils.logging import logger


def create_temp_file(content: bytes, filename: str) -> Path:

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    temp_file_path = temp_dir / filename
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(content)

    return temp_file_path


def clean_temp_files(temp_file_path: Optional[Path] = None):
    """
    Clean temporary files

    Args:
        temp_file_path: Specific file to clean, if None clean all temp files
    """
    try:
        if temp_file_path and temp_file_path.exists():
            os.remove(temp_file_path)
            logger.debug(f"Removed temporary file: {temp_file_path}")
        elif not temp_file_path:
            temp_dir = Path("temp")
            if temp_dir.exists():
                for file in temp_dir.iterdir():
                    if file.is_file():
                        os.remove(file)
                logger.debug("Cleaned all temporary files")
    except Exception as e:
        logger.error(f"Error cleaning temporary files: {str(e)}")


def get_extension(filename: str) -> str:
    """
    Get file extension from filename

    Args:
        filename: Name of the file

    Returns:
        File extension without the dot (e.g. 'txt')
    """
    return filename.split(".")[-1] if "." in filename else ""


def is_supported_file_type(filename: str, supported_types: List[str] = None) -> bool:
    """
    Check if the file type is supported

    Args:
        filename: Name of the file
        supported_types: List of supported file extensions without dot
                        (defaults to ['txt'])

    Returns:
        True if file type is supported, False otherwise
    """
    if supported_types is None:
        supported_types = ["txt"]

    extension = get_extension(filename)
    return extension.lower() in [t.lower() for t in supported_types]


def format_query_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the query result for the API response

    Args:
        result: Raw query result

    Returns:
        Formatted result
    """
    # Filter out sensitive or unnecessary information
    formatted = {
        "query": result.get("query", ""),
        "intent": result.get("intent", ""),
        "slots": result.get("slots", {}),
        "is_out_of_scope": result.get("is_out_of_scope", False),
        "response": result.get("response", ""),
        "confidence": result.get("confidence", 0.0),
    }

    # Format documents used
    if "documents_used" in result and result["documents_used"]:
        formatted["documents_used"] = [
            doc if isinstance(doc, str) else doc.content if hasattr(doc, "content") else str(doc)
            for doc in result["documents_used"]
        ]
    else:
        formatted["documents_used"] = []

    return formatted