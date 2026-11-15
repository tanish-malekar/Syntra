"""The backend serves the built website: /api/* is the API, page loads get index.html, API clients are unaffected."""
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_website_and_api_share_one_address(tmp_path):
    web = tmp_path / "dist"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_text("<!doctype html><title>Syntra</title>")
    (web / "assets" / "app.js").write_text("console.log(1)")
    app = create_app(Settings(database_url="sqlite://", run_background_tasks=False, web_dir=str(web)))
    with TestClient(app) as c:
        html = {"accept": "text/html,application/xhtml+xml"}
        assert c.get("/", headers=html).text.startswith("<!doctype html>")
        assert c.get("/models", headers=html).text.startswith("<!doctype html>")     # a page path that is also an API path
        assert c.get("/models").status_code == 401                                    # API clients still reach the API
        assert c.get("/api/models").status_code == 401 and c.get("/api/health").json() == {"status": "ok"}
        js = c.get("/assets/app.js")
        assert js.text == "console.log(1)" and "javascript" in js.headers["content-type"] and "immutable" in js.headers["cache-control"]
        assert c.get("/assets/../../etc/passwd", headers=html).text.startswith("<!doctype html>")  # never leaves dist
        assert c.get("/health").json() == {"status": "ok"}


def test_without_a_build_the_backend_is_api_only(tmp_path):
    app = create_app(Settings(database_url="sqlite://", run_background_tasks=False, web_dir=str(tmp_path / "missing")))
    with TestClient(app) as c:
        assert c.get("/", headers={"accept": "text/html"}).status_code == 404
