"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
# import pyttsx3

from callback import StreamingLLMCallbackHandler
from chain import get_chain
from schemas import ChatResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# engine = pyttsx3.init()

# @app.on_event("startup")
# async def startup_event():
#     logging.info("loading vectorstore")
#     if not Path("vectorstore.pkl").exists():
#         raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
#     with open("vectorstore.pkl", "rb") as f:
#         global vectorstore
#         vectorstore = pickle.load(f)


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chain = get_chain(stream_handler=stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            # Get and speak the result
            result = await chain.apredict(input=question)
            # engine.say(result)
            # engine.runAndWait()

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())

        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break

        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())

    # Stop the pyttsx3 engine
    # engine.stop()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
