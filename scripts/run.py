import argparse
import uvicorn


def main():
    """Run the FastAPI application"""
    parser = argparse.ArgumentParser(description="Document Chatbot API")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the API on"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the API on"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot reload for development"
    )
    args = parser.parse_args()

    # Initialize the application
    try:
        from utils.initialize import initialize_app
        initialize_app()
    except Exception as e:
        return

    # Run the application
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()