import pytest

from app.auth.models import AccessLevel, Permission, Principal, Role
from app.auth.users import UserDirectory
from app.core.config import Settings
from app.core.exceptions import AuthenticationError
from app.core.security import create_access_token, decode_access_token, hash_password, verify_password

# The RBAC matrix from CLAUDE.md §8, written out literally so any drift fails loudly.
EXPECTED_MATRIX = {
    Role.VIEWER: {"chat", "search"},
    Role.ANALYST: {"chat", "search", "analytics", "mcp"},
    Role.ADMINISTRATOR: {"chat", "search", "analytics", "mcp", "admin"},
}


def _principal(role: Role) -> Principal:
    return Principal(user_id="u", username="u", display_name="U", role=role)


@pytest.mark.parametrize("role", list(Role))
def test_role_permissions_match_claude_md_matrix(role):
    assert {p.value for p in _principal(role).permissions} == EXPECTED_MATRIX[role]


def test_clearance_is_monotonic_by_role():
    viewer, analyst, admin = (_principal(r) for r in Role)
    assert viewer.allowed_access_levels == (AccessLevel.PUBLIC, AccessLevel.INTERNAL)
    assert analyst.can_read("confidential") and not analyst.can_read("restricted")
    assert admin.can_read("restricted")
    assert not viewer.can_read("confidential")


def test_viewer_cannot_use_tools():
    viewer = _principal(Role.VIEWER)
    assert not viewer.has(Permission.ANALYTICS)
    assert not viewer.has(Permission.MCP)
    assert not viewer.has(Permission.ADMIN)


def test_password_hash_roundtrip_and_rejects_tampering():
    encoded = hash_password("s3cret")
    assert verify_password("s3cret", encoded)
    assert not verify_password("wrong", encoded)
    assert not verify_password("s3cret", "md5$abc")


async def test_directory_authenticates_demo_users():
    directory = UserDirectory()
    principal = await directory.authenticate("analyst", "analyst-demo-pass")
    assert principal is not None and principal.role is Role.ANALYST
    assert await directory.authenticate("analyst", "nope") is None
    assert await directory.authenticate("ghost", "analyst-demo-pass") is None


def test_token_roundtrip_and_signature_enforced():
    settings = Settings(jwt_secret="a" * 40)
    token, ttl = create_access_token(subject="u-2001", role="analyst", settings=settings)
    assert ttl == settings.jwt_ttl_minutes * 60
    assert decode_access_token(token, settings)["sub"] == "u-2001"

    with pytest.raises(AuthenticationError):
        decode_access_token(token, Settings(jwt_secret="b" * 40))


def test_expired_token_rejected():
    settings = Settings(jwt_secret="a" * 40, jwt_ttl_minutes=-1)
    token, _ = create_access_token(subject="u-2001", role="analyst", settings=settings)
    with pytest.raises(AuthenticationError, match="expired"):
        decode_access_token(token, settings)


def test_production_refuses_default_secret():
    with pytest.raises(ValueError, match="JWT_SECRET"):
        Settings(app_env="production", redis_url="redis://x")
