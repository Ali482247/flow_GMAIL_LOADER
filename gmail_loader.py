import base64
import json
import os
import re
from typing import Iterator, Any

from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage
from langflow.custom import Component
from langflow.inputs import MessageTextInput
from langflow.schema import Data
from langflow.template import Output
from loguru import logger


class GmailOAuthLoaderComponent(Component):
    display_name = "Gmail OAuth Loader"
    description = "Loads emails from Gmail using OAuth 2.0 user credentials."
    icon = "🔐"

    inputs = [
        MessageTextInput(
            name="client_secret_path",
            display_name="OAuth Client Secret Path",
            required=True,
            value="/home/aliakbar7887/my_projects/langflow/client_secret.json",
        ),
        MessageTextInput(
            name="label_ids",
            display_name="Label IDs",
            required=False,
            value="INBOX",
        ),
        MessageTextInput(
            name="max_results",
            display_name="Max Emails",
            required=False,
            value="10",
        ),
    ]

    outputs = [
        Output(display_name="Data", name="data", method="load_emails"),
    ]

    def clean_content(self, text: str) -> str:
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"[^A-Za-z0-9\s]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def extract_text_from_parts(self, parts: list) -> str:
        for part in parts:
            mime_type = part.get("mimeType")
            body = part.get("body", {})
            data = body.get("data")

            if mime_type == "text/plain" and data:
                decoded = base64.urlsafe_b64decode(data).decode("utf-8")
                return self.clean_content(decoded)

            elif mime_type == "text/html" and data:
                decoded = base64.urlsafe_b64decode(data).decode("utf-8")
                soup = BeautifulSoup(decoded, "html.parser")
                return self.clean_content(soup.get_text())

            elif part.get("parts"):
                result = self.extract_text_from_parts(part["parts"])
                if result:
                    return result

        return ""

    def get_plain_text(self, msg: dict) -> str:
        payload = msg.get("payload", {})
        if "parts" in payload:
            return self.extract_text_from_parts(payload["parts"])
        elif "body" in payload and "data" in payload["body"]:
            decoded = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")
            return self.clean_content(decoded)
        return ""

    def load_emails(self) -> Data:
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        client_secret_path = self.client_secret_path
        token_path = os.path.join(os.path.dirname(client_secret_path), "token.json")

        creds = None
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(token_path, 'w') as token:
                token.write(creds.to_json())

        try:
            service = build("gmail", "v1", credentials=creds)
        except Exception as e:
            raise ValueError(f"Failed to initialize Gmail API service: {e}")

        label_ids = self.label_ids.split(",") if self.label_ids else ["INBOX"]
        max_results = int(self.max_results) if self.max_results else 10

        try:
            results = service.users().messages().list(
                userId="me", labelIds=label_ids, maxResults=max_results
            ).execute()
        except Exception as e:
            raise ValueError(f"Failed to fetch messages list: {e}")

        messages = results.get("messages", [])
        if not messages:
            logger.warning("No messages found with the specified labels.")

        output = []
        for msg in messages:
            try:
                full_msg = service.users().messages().get(userId="me", id=msg["id"]).execute()
                sender = next((h["value"] for h in full_msg["payload"]["headers"]
                               if h["name"].lower() == "from"), "unknown")
                content = self.get_plain_text(full_msg)
                if content:
                    output.append(ChatSession(messages=[
                        HumanMessage(content=content, additional_kwargs={"sender": sender})
                    ]))
            except Exception as e:
                logger.exception(f"Failed to process message {msg['id']}: {e}")

        return Data(data={"text": output})
