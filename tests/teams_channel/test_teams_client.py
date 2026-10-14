"""Behaviour tests for teams_client, driven over a real local HTTP fake.

Every assertion is about what the module sends and does: the request sequence,
the member roster it builds, the skype-token header, where it keeps tokens and
with what permissions, and that an expired access token refreshes without a
second sign-in. Nothing here stubs the module's own functions.
"""

import os
import stat
from argparse import Namespace

import pytest

import teams_client
from teams_client import TeamsClient, TeamsError


def _signin(client):
    """Sign in through the fake with no real polling delay or console output."""
    client.signin(interval=0, prompt=lambda _m: None)


# -- sign-in and status ----------------------------------------------------


def test_signin_stores_token_and_status_names_account(fake, capsys):
    client, state, _root = fake
    _signin(client)

    assert state.events[:2] == ["devicecode", "token:device"]
    assert teams_client._cmd_status(client, Namespace()) == 0
    out = capsys.readouterr().out
    assert "Li Rui" in out
    assert "lirui927@contoso.com" in out


def test_me_decodes_the_account_from_the_token(fake):
    client, _state, _root = fake
    _signin(client)
    who = client.me()
    assert who == {
        "oid": "oid-abc-123",
        "username": "lirui927@contoso.com",
        "name": "Li Rui",
    }


def test_status_fails_before_sign_in(tmp_path, capsys):
    client = TeamsClient(state_dir=tmp_path / "state")
    with pytest.raises(TeamsError):
        client.me()
    assert teams_client._cmd_status(client, Namespace()) == 1


# -- the named result: one full round trip ---------------------------------


def test_create_send_list_round_trip_in_order(fake):
    client, state, _root = fake
    _signin(client)

    chat_id = client.create_chat("teams-channel test")
    client.send_message(chat_id, "hello from the fake")
    messages = client.list_messages(chat_id)

    assert [m["content"] for m in messages] == ["hello from the fake"]
    assert state.events == [
        "devicecode", "token:device", "authz", "create", "send", "list",
    ]


# -- create_chat shape -----------------------------------------------------


def test_create_chat_posts_a_self_only_roster(fake):
    client, state, _root = fake
    _signin(client)

    chat_id = client.create_chat("standup")
    assert chat_id == state.next_chat_id

    create = next(r for r in state.requests if r["label"] == "create")
    assert create["body"]["members"] == [
        {"id": "8:orgid:oid-abc-123", "role": "Admin"}
    ]
    assert create["body"]["properties"] == {"threadType": "chat", "topic": "standup"}


def test_create_chat_cli_prints_the_new_id(fake, capsys):
    client, state, _root = fake
    _signin(client)
    assert teams_client._cmd_create_chat(client, Namespace(topic="t")) == 0
    assert capsys.readouterr().out.strip() == state.next_chat_id


# -- message shape and headers ---------------------------------------------


def test_send_message_marks_plain_text(fake):
    client, state, _root = fake
    _signin(client)
    chat_id = client.create_chat("t")
    client.send_message(chat_id, "hi")

    send = next(r for r in state.requests if r["label"] == "send")
    assert send["body"]["messagetype"] == "Text"
    assert send["body"]["contenttype"] == "text"
    assert send["body"]["content"] == "hi"
    assert send["body"]["clientmessageid"]  # a dedup id is always stamped


def test_chat_calls_carry_the_skype_token_header(fake):
    client, state, _root = fake
    _signin(client)
    client.list_messages("19:somechat@thread.v2")

    listed = next(r for r in state.requests if r["label"] == "list")
    assert listed["headers"]["Authentication"] == "skypetoken=SKYPE-TOKEN"


def test_list_messages_passes_page_size(fake):
    client, state, _root = fake
    _signin(client)
    client.list_messages("19:somechat@thread.v2", page_size=7)
    listed = next(r for r in state.requests if r["label"] == "list")
    assert listed["body"]["pageSize"] == ["7"]


# -- token refresh ---------------------------------------------------------


def test_expired_access_token_refreshes_without_a_second_signin(fake):
    client, state, _root = fake
    state.device_expires_in = 0  # the sign-in token is already stale
    _signin(client)

    # Any authenticated call must mint a fresh token from the refresh token.
    client.me()

    assert state.refresh_count == 1
    assert state.events.count("devicecode") == 1
    assert "token:refresh" in state.events


def test_device_code_polling_waits_for_pending(fake):
    client, state, _root = fake
    state.pending_polls = 2
    _signin(client)

    assert state.events.count("token:pending") == 2
    assert client.me()["name"] == "Li Rui"


# -- token storage ---------------------------------------------------------


def test_tokens_are_written_owner_only(fake):
    client, _state, _root = fake
    _signin(client)
    client.create_chat("t")  # forces the skype token to be minted and saved

    state_dir = client.state_dir
    assert stat.S_IMODE(os.stat(state_dir).st_mode) == 0o700
    for name in ("token.json", "skypetoken.json"):
        path = state_dir / name
        assert path.exists()
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_state_is_confined_to_the_state_dir(fake, tmp_path):
    client, _state, root = fake
    _signin(client)

    # A client on a different state dir shares nothing: tokens come only from
    # the state directory, never from the environment or a neighbour's path.
    other = TeamsClient(state_dir=tmp_path / "elsewhere", login_base=root)
    with pytest.raises(TeamsError):
        other.me()
