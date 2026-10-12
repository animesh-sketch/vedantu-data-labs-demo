"""Connector architecture.

Every source implements the same small interface. Status is derived from real configuration only:
a connector is never shown as "Connected" unless it actually loaded data in this session.
Secrets are read from environment variables (.env, never committed) and never displayed.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

NOT_CONFIGURED = "Not configured"
CONFIGURED = "Credentials present – not verified"
LOCAL = "Available (local)"


@dataclass
class ConnectorInfo:
    key: str
    name: str
    auth: str
    env_vars: list[str]
    scopes: str
    status: str
    notes: str


class Connector(ABC):
    key: str = ""
    name: str = ""
    auth: str = ""
    env_vars: list[str] = []
    scopes: str = ""
    notes: str = ""

    def is_configured(self) -> bool:
        return bool(self.env_vars) and all(os.getenv(v) for v in self.env_vars)

    def status(self) -> str:
        return CONFIGURED if self.is_configured() else NOT_CONFIGURED

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(self.key, self.name, self.auth, self.env_vars, self.scopes, self.status(), self.notes)

    @abstractmethod
    def fetch(self, obj: str, limit: int = 1000) -> pd.DataFrame: ...


class FileConnector(Connector):
    key, name, auth, scopes = "file", "CSV / Excel upload", "None (local file)", "n/a"
    notes = "Files are processed in memory for this session only."

    def status(self) -> str:
        return LOCAL

    def fetch(self, obj: str, limit: int = 1000) -> pd.DataFrame:
        from .profiling import load_file
        return load_file(obj, obj).head(limit)


class _OAuthCRM(Connector):
    def fetch(self, obj: str, limit: int = 1000) -> pd.DataFrame:
        if not self.is_configured():
            raise ConnectionError(f"{self.name} is not configured. Set {', '.join(self.env_vars)} in .env.")
        raise NotImplementedError(
            f"{self.name} fetch is a documented extension point. Implement the OAuth token exchange and a "
            f"read-only paginated GET for '{obj}' here, then map records to a DataFrame.")


class HubSpotConnector(_OAuthCRM):
    key, name = "hubspot", "HubSpot CRM"
    auth = "OAuth 2.0 (authorization code) or Private App token"
    env_vars = ["HUBSPOT_CLIENT_ID", "HUBSPOT_CLIENT_SECRET", "HUBSPOT_REDIRECT_URI"]
    scopes = "crm.objects.contacts.read, crm.objects.deals.read (read-only)"
    notes = "Endpoint: GET /crm/v3/objects/{object}?limit=100&after=… (cursor pagination, 100 req/10s)."


class SalesforceConnector(_OAuthCRM):
    key, name = "salesforce", "Salesforce"
    auth = "OAuth 2.0 (web server flow / JWT bearer for service users)"
    env_vars = ["SALESFORCE_CLIENT_ID", "SALESFORCE_CLIENT_SECRET", "SALESFORCE_INSTANCE_URL"]
    scopes = "api, refresh_token (read-only integration user profile)"
    notes = "Endpoint: GET /services/data/vXX.X/query?q=SELECT … (SOQL, read-only profile)."


class PostgresConnector(_OAuthCRM):
    key, name = "postgres", "PostgreSQL / Warehouse"
    auth = "Read-only database role over TLS"
    env_vars = ["VDL_DB_URL"]
    scopes = "SELECT on approved analytics schemas only"
    notes = "Use a read-only replica; queries still pass through the same SQL guard."


REGISTRY: list[Connector] = [FileConnector(), HubSpotConnector(), SalesforceConnector(), PostgresConnector()]


def all_info() -> list[ConnectorInfo]:
    return [c.info() for c in REGISTRY]


def get(key: str) -> Connector:
    for c in REGISTRY:
        if c.key == key:
            return c
    raise KeyError(key)
