#!/usr/bin/env python3
"""
Generate a single HTML page from dupes.csv (or groups.csv) for moderation review.
Each row/group is shown as a section with animations side-by-side: thumbnail + name + catalog link.
Fetches thumbnail URLs and asset names from Roblox APIs at build time.
"""
import argparse
import csv
import html as html_module
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler


CATALOG_URL = "https://www.roblox.com/catalog/{id}"
THUMBNAIL_API = "https://thumbnails.roblox.com/v1/assets"
# Catalog page HTML has <title>Name - Roblox</title>; economy API is rate-limited
CATALOG_PAGE = "https://www.roblox.com/catalog/{id}"
BATCH_SIZE = 50  # Roblox may limit how many per request
REQUEST_DELAY = 0.3  # seconds between thumbnail batch calls
ECONOMY_DELAY = 1.2  # seconds between economy API calls (strict rate limit; skip with --no-names)


def load_dupes_format(path: str) -> list[list[str]]:
    """No header: each row = one group, comma-separated anim IDs."""
    groups = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            ids = [c.strip() for c in row if c.strip()]
            if ids:
                groups.append(ids)
    return groups


def load_groups_format(path: str) -> list[list[str]]:
    """Header animId, clipId, duration, groupId; aggregate by groupId, return list of animIds per group (sorted)."""
    by_group = defaultdict(list)
    seen_gids = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "groupId" not in (reader.fieldnames or []):
            return []
        for row in reader:
            anim_id = (row.get("animId") or "").strip()
            if not anim_id:
                continue
            try:
                gid = int(row.get("groupId", 0))
            except (ValueError, TypeError):
                gid = 0
            by_group[gid].append(anim_id)
            if gid not in seen_gids:
                seen_gids.append(gid)
    groups = []
    for gid in seen_gids:
        ids = sorted(set(by_group[gid]), key=lambda x: (int(x) if x.isdigit() else 0, x))
        if ids:
            groups.append(ids)
    return groups


def detect_and_load(path: str) -> list[list[str]]:
    """Detect format and return list of groups (each group = list of anim IDs)."""
    with open(path, newline="", encoding="utf-8") as f:
        first = f.readline()
    # Check for groups.csv header
    if "animId" in first and "groupId" in first:
        groups = load_groups_format(path)
        if groups:
            return groups
    return load_dupes_format(path)


def escape(s: str) -> str:
    return html_module.escape(s, quote=True)


def fetch_thumbnails(asset_ids: list[str], verbose: bool = False) -> dict[str, str]:
    """Return map asset_id -> imageUrl from thumbnails.roblox.com (batched)."""
    result = {}
    numeric_ids = [aid for aid in asset_ids if aid.isdigit()]
    for i in range(0, len(numeric_ids), BATCH_SIZE):
        batch = numeric_ids[i : i + BATCH_SIZE]
        # API expects assetIds as comma-separated; size must be a documented value (250x250, 512x512, etc.)
        ids_param = ",".join(batch)
        url = f"{THUMBNAIL_API}?assetIds={ids_param}&size=250x250&format=Png&isCircular=false"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
            for item in data.get("data", []):
                aid = str(item.get("targetId", ""))
                if item.get("state") == "Completed" and item.get("imageUrl"):
                    result[aid] = item["imageUrl"]
                elif verbose and aid:
                    print(f"  thumb state for {aid}: {item.get('state')}")
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            if verbose:
                print(f"  thumbnails HTTP {e.code}: {body[:200]}")
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            if verbose:
                print(f"  thumbnails error: {e}")
        time.sleep(REQUEST_DELAY)
    return result


def fetch_asset_names(asset_ids: list[str], verbose: bool = False) -> dict[str, str]:
    """Return map asset_id -> name from catalog page <title> (e.g. 'Name - Roblox')."""
    result = {}
    numeric_ids = [aid for aid in asset_ids if aid.isdigit()]
    if numeric_ids:
        time.sleep(REQUEST_DELAY)  # pause after thumbnail calls
    for aid in numeric_ids:
        try:
            url = CATALOG_PAGE.format(id=aid)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                page_html = resp.read().decode(errors="replace")
            m = re.search(r"<title>([^<]+)</title>", page_html, re.IGNORECASE)
            if m:
                title = m.group(1).strip()
                # Strip " - Roblox" or " | Roblox" suffix
                for suffix in (" - Roblox", " | Roblox"):
                    if title.endswith(suffix):
                        title = title[: -len(suffix)].strip()
                        break
                if title:
                    result[aid] = title
                    continue
            result[aid] = f"Asset {aid}"
        except urllib.error.HTTPError as e:
            if verbose and e.code != 404:
                print(f"  catalog page {aid} HTTP {e.code}")
            result[aid] = f"Asset {aid}"
        except (urllib.error.URLError, OSError) as e:
            if verbose:
                print(f"  catalog page {aid} error: {e}")
            result[aid] = f"Asset {aid}"
        time.sleep(ECONOMY_DELAY)
    return result


def render_html(
    groups: list[list[str]],
    output_path: str,
    thumb_urls: dict[str, str],
    asset_names: dict[str, str],
) -> None:
    """Write a single self-contained HTML file."""
    sections_html = []
    for idx, anim_ids in enumerate(groups, start=1):
        if not anim_ids:
            continue
        cards = []
        for aid in anim_ids:
            catalog_url = CATALOG_URL.format(id=escape(aid))
            thumb_url = thumb_urls.get(aid)
            name = asset_names.get(aid) or f"Asset {escape(aid)}"
            if thumb_url:
                img_html = f'<img src="{escape(thumb_url)}" alt="{escape(name)}" loading="lazy" onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'block\';">'
            else:
                img_html = '<img style="display:none;"><span class="placeholder" style="display:block;">[No thumbnail]</span>'
            cards.append(
                f"""<div class="card">
  <a href="{catalog_url}" target="_blank" rel="noopener">
    {img_html}
  </a>
  <div class="meta">
    <span class="name">{escape(name)}</span>
    <span class="id">{escape(aid)}</span>
    <a href="{catalog_url}" target="_blank" rel="noopener" class="link">View on Roblox</a>
  </div>
</div>"""
            )
        catalog_urls_js = ",".join(repr(CATALOG_URL.format(id=aid)) for aid in anim_ids)
        open_all_btn = f'<button type="button" class="open-all" onclick="openInSideFrame([{catalog_urls_js}])">Open in side panel</button>'
        sections_html.append(
            f"""<section class="group">
  <h2>Group {idx} ({len(anim_ids)} animation{"s" if len(anim_ids) != 1 else ""})</h2>
  {open_all_btn}
  <div class="cards">
    {"".join(cards)}
  </div>
</section>"""
        )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dupe groups review</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, sans-serif; margin: 0; padding: 1rem 2rem; background: #1a1a1a; color: #e0e0e0; }}
    h1 {{ margin-bottom: 0.5rem; }}
    .sub {{ color: #888; margin-bottom: 1.5rem; }}
    .group {{ margin-bottom: 2.5rem; padding: 1rem; background: #252525; border-radius: 8px; }}
    .group h2 {{ margin: 0 0 0.75rem 0; font-size: 1.1rem; }}
    .open-all {{ margin-bottom: 0.75rem; padding: 0.4rem 0.8rem; cursor: pointer; background: #0e639c; color: #fff; border: none; border-radius: 4px; font-size: 0.9rem; }}
    .open-all:hover {{ background: #1177bb; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
    .card {{ width: 180px; background: #333; border-radius: 6px; overflow: hidden; }}
    .card a {{ color: #6af; text-decoration: none; }}
    .card a:hover {{ text-decoration: underline; }}
    .card img {{ display: block; width: 100%; aspect-ratio: 1; object-fit: cover; }}
    .card .placeholder {{ display: block; width: 100%; aspect-ratio: 1; background: #444; color: #888; text-align: center; line-height: 180px; }}
    .card .meta {{ padding: 0.5rem; }}
    .card .name {{ display: block; font-size: 0.9rem; color: #e0e0e0; margin-bottom: 0.2rem; line-height: 1.2; max-height: 2.4em; overflow: hidden; text-overflow: ellipsis; }}
    .card .id {{ font-family: monospace; font-size: 0.8rem; color: #888; }}
    .card .link {{ display: inline-block; margin-top: 0.25rem; font-size: 0.85rem; }}
    .side-frame {{ position: fixed; top: 0; right: 0; width: min(55%, 960px); height: 100%; background: #111; z-index: 1000; display: none; flex-direction: column; box-shadow: -4px 0 12px rgba(0,0,0,0.5); }}
    .side-frame.visible {{ display: flex; }}
    .side-frame .header {{ padding: 0.5rem 1rem; background: #252525; display: flex; align-items: center; gap: 0.5rem; flex-shrink: 0; }}
    .side-frame .header button {{ padding: 0.4rem 0.8rem; cursor: pointer; background: #0e639c; color: #fff; border: none; border-radius: 4px; font-size: 0.9rem; }}
    .side-frame .header button:hover {{ background: #1177bb; }}
    .side-frame .header .close {{ margin-left: auto; }}
    .side-frame .panel-body {{ flex: 1; overflow: auto; padding: 1rem; }}
    .side-frame .panel-body p {{ color: #888; font-size: 0.9rem; margin: 0 0 0.5rem 0; }}
    .side-frame .link-list {{ list-style: none; padding: 0; margin: 0; }}
    .side-frame .link-list li {{ margin-bottom: 0.4rem; }}
    .side-frame .link-list a {{ color: #6af; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>Dupe groups review</h1>
  <p class="sub">{len(groups)} group(s)</p>
  {"".join(sections_html)}
  <div id="sideFrame" class="side-frame">
    <div class="header">
      <button type="button" class="close" id="sideFrameClose">Close</button>
    </div>
    <div class="panel-body">
      <p>Open each in a new tab:</p>
      <ul id="sideLinkList" class="link-list"></ul>
      <button type="button" id="openAllNewTabs" style="margin-top:0.75rem; padding:0.4rem 0.8rem; background:#0e639c; color:#fff; border:none; border-radius:4px; cursor:pointer;">Open all in new tabs</button>
    </div>
  </div>
  <script>
    (function() {{
      var urls = [];
      var panel = document.getElementById('sideFrame');
      var listEl = document.getElementById('sideLinkList');
      window.openInSideFrame = function(urlList) {{
        urls = urlList.slice();
        listEl.innerHTML = '';
        urlList.forEach(function(url, i) {{
          var li = document.createElement('li');
          var a = document.createElement('a');
          a.href = url;
          a.target = '_blank';
          a.rel = 'noopener';
          a.textContent = 'Asset ' + (i + 1);
          li.appendChild(a);
          listEl.appendChild(li);
        }});
        panel.classList.add('visible');
      }};
      document.getElementById('openAllNewTabs').onclick = function() {{
        var delay = 0;
        urls.forEach(function(url) {{
          setTimeout(function() {{ window.open(url, '_blank', 'noopener'); }}, delay);
          delay += 300;
        }});
      }};
      document.getElementById('sideFrameClose').onclick = function() {{ panel.classList.remove('visible'); }};
    }})();
  </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def _find_clip_csv_for_asset(asset_id: str, train_data_dir: str) -> str | None:
    """Return path to a clip CSV for this asset (animId), or None. Uses first match of {asset_id}-*.csv."""
    if not asset_id or not asset_id.isdigit():
        return None
    try:
        candidates = [
            os.path.join(train_data_dir, f)
            for f in os.listdir(train_data_dir)
            if f.startswith(asset_id + "-") and f.endswith(".csv") and f != "manifest.csv"
        ]
        return sorted(candidates)[0] if candidates else None
    except OSError:
        return None


def _render_skeleton_gif(asset_id: str, train_data_dir: str, cache_dir: str, fps: int = 12, dpi: int = 60) -> str | None:
    """Find clip CSV for asset, render to cache_dir/{asset_id}.gif, return path or None. Lazy-imports visualize_skeleton."""
    cache_path = os.path.join(cache_dir, f"{asset_id}.gif")
    if os.path.isfile(cache_path):
        return cache_path
    clip_path = _find_clip_csv_for_asset(asset_id, train_data_dir)
    if not clip_path:
        return None
    try:
        _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _repo not in sys.path:
            sys.path.insert(0, _repo)
        from ml_training.visualize_skeleton import render_one_gif
        os.makedirs(cache_dir, exist_ok=True)
        if render_one_gif(clip_path, cache_path, fps=fps, dpi=dpi, side_views=True):
            return cache_path
    except Exception:
        pass
    return None


def _make_proxy_handler(serve_dir: str, train_data_dir: str, skeleton_cache_dir: str) -> type:
    """Build a request handler that serves static files and proxies /api/thumbnails, /api/names, /api/skeleton_gif."""

    class _ProxyHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

            if path == "/api/skeleton_gif" and "asset_id" in qs:
                aid = (qs["asset_id"][0] or "").strip()
                if not aid:
                    self.send_error(400)
                    return
                gif_path = _render_skeleton_gif(aid, train_data_dir, skeleton_cache_dir)
                if gif_path is None:
                    self.send_error(404)
                    return
                try:
                    with open(gif_path, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/gif")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                except OSError:
                    self.send_error(404)
                return

            if path == "/api/thumbnails" and "ids" in qs:
                ids = [x.strip() for x in qs["ids"][0].split(",") if x.strip()]
                try:
                    result = fetch_thumbnails(ids)
                    self._send_json(200, {"data": [{"targetId": int(k), "imageUrl": v, "state": "Completed"} for k, v in result.items() if k.isdigit()]})
                except Exception as e:
                    self._send_json(500, {"error": str(e)})
                return

            if path == "/api/names" and "ids" in qs:
                ids = [x.strip() for x in qs["ids"][0].split(",") if x.strip()]
                try:
                    result = fetch_asset_names(ids)
                    self._send_json(200, result)
                except Exception as e:
                    self._send_json(500, {"error": str(e)})
                return

            # Serve static file
            if path == "/" or path == "":
                path = "/dupes_review.html"
            file_path = os.path.join(serve_dir, path.lstrip("/"))
            if not os.path.isfile(file_path):
                self.send_error(404)
                return
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                ext = os.path.splitext(file_path)[1].lower()
                ctype = "text/html" if ext == ".html" else "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except OSError:
                self.send_error(404)

        def _send_json(self, status: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return _ProxyHandler


def serve_lazy(
    output_path: str,
    port: int = 8765,
    train_data_dir: str | None = None,
    skeleton_cache_dir: str | None = None,
) -> None:
    """Serve the lazy HTML and proxy API from the same directory as output_path."""
    serve_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_script_dir)
    if train_data_dir is None:
        train_data_dir = os.path.join(_repo_root, "ml_training", "train_data")
    if skeleton_cache_dir is None:
        skeleton_cache_dir = os.path.join(_repo_root, "skeleton_gifs")
    handler = _make_proxy_handler(serve_dir, train_data_dir, skeleton_cache_dir)
    with HTTPServer(("", port), handler) as httpd:
        base = f"http://127.0.0.1:{port}"
        print(f"Serving at {base}/")
        print(f"Open {base}/dupes_review.html then click 'Load in side panel' on a group.")
        httpd.serve_forever()


def render_html_lazy(groups: list[list[str]], output_path: str, use_proxy: bool = False) -> None:
    """Write HTML that shows only group IDs; thumbnails and names load in side panel on demand.
    When use_proxy is True, the page expects to be served from the same host with /api/thumbnails and /api/names (run with --serve).
    """
    sections_html = []
    for idx, anim_ids in enumerate(groups, start=1):
        if not anim_ids:
            continue
        ids_display = ", ".join(escape(aid) for aid in anim_ids)
        ids_attr = escape(json.dumps(anim_ids))
        sections_html.append(
            f"""<section class="group lazy-group" data-group-index="{idx - 1}">
  <h2>Group {idx} ({len(anim_ids)} animation{"s" if len(anim_ids) != 1 else ""})</h2>
  <p class="ids">{ids_display}</p>
  <button type="button" class="open-all load-in-panel" data-ids="{ids_attr}">Load in side panel</button>
</section>"""
        )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dupe groups review (on-demand)</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, sans-serif; margin: 0; padding: 1rem 2rem; background: #1a1a1a; color: #e0e0e0; }}
    h1 {{ margin-bottom: 0.5rem; }}
    .sub {{ color: #888; margin-bottom: 1.5rem; }}
    .group {{ margin-bottom: 1.5rem; padding: 1rem; background: #252525; border-radius: 8px; }}
    .group h2 {{ margin: 0 0 0.5rem 0; font-size: 1.1rem; }}
    .group .ids {{ font-family: monospace; font-size: 0.85rem; color: #aaa; word-break: break-all; margin: 0 0 0.75rem 0; }}
    .open-all {{ padding: 0.4rem 0.8rem; cursor: pointer; background: #0e639c; color: #fff; border: none; border-radius: 4px; font-size: 0.9rem; }}
    .open-all:hover {{ background: #1177bb; }}
    .side-frame {{ position: fixed; top: 0; right: 0; width: min(58%, 1000px); height: 100%; background: #111; z-index: 1000; display: none; flex-direction: column; box-shadow: -4px 0 12px rgba(0,0,0,0.5); }}
    .side-frame.visible {{ display: flex; }}
    .side-frame .header {{ padding: 0.5rem 1rem; background: #252525; display: flex; align-items: center; gap: 0.5rem; flex-shrink: 0; }}
    .side-frame .header button {{ padding: 0.4rem 0.8rem; cursor: pointer; background: #0e639c; color: #fff; border: none; border-radius: 4px; font-size: 0.9rem; }}
    .side-frame .header .close {{ margin-left: auto; }}
    .side-frame .panel-body {{ flex: 1; overflow: auto; padding: 1rem; }}
    .side-frame .loading {{ color: #888; }}
    .side-frame .cards {{ display: flex; flex-wrap: wrap; gap: 1.25rem; }}
    .side-frame .card {{ width: 320px; background: #333; border-radius: 8px; overflow: hidden; }}
    .side-frame .card a {{ color: #6af; text-decoration: none; font-size: 0.95rem; }}
    .side-frame .card img {{ display: block; width: 100%; aspect-ratio: 1; object-fit: cover; }}
    .side-frame .card .placeholder {{ display: block; width: 100%; aspect-ratio: 1; background: #444; color: #666; text-align: center; font-size: 0.85rem; line-height: 320px; }}
    .side-frame .card .meta {{ padding: 0.5rem; }}
    .side-frame .card .name {{ display: block; font-size: 0.95rem; color: #e0e0e0; line-height: 1.2; max-height: 2.6em; overflow: hidden; text-overflow: ellipsis; }}
    .side-frame .card .id {{ font-family: monospace; font-size: 0.8rem; color: #888; }}
    .side-frame .link-list {{ list-style: none; padding: 0; margin: 0.5rem 0 0 0; }}
    .side-frame .link-list li {{ margin-bottom: 0.3rem; }}
    .side-frame .link-list a {{ color: #6af; font-size: 0.9rem; }}
    .side-frame .card .skeleton-gif {{ display: block; width: 100%; max-height: 320px; min-height: 200px; object-fit: contain; background: #222; }}
  </style>
</head>
<body>
  <h1>Dupe groups review (on-demand)</h1>
  <p class="sub">{len(groups)} group(s) — click "Load in side panel" to fetch thumbnails and names</p>
  {"".join(sections_html)}
  <div id="sideFrame" class="side-frame">
    <div class="header">
      <button type="button" class="close" id="sideFrameClose">Close</button>
    </div>
    <div class="panel-body" id="sidePanelBody">
      <p class="loading">Select a group to load.</p>
    </div>
  </div>
  <script>
    (function() {{
      var USE_PROXY = {str(use_proxy).lower()};
      var THUMB_API = USE_PROXY ? "/api/thumbnails" : "https://thumbnails.roblox.com/v1/assets";
      var CATALOG_URL = "https://www.roblox.com/catalog/";
      var BATCH = 50;
      var panel = document.getElementById('sideFrame');
      var bodyEl = document.getElementById('sidePanelBody');

      function esc(s) {{
        var div = document.createElement('div');
        div.textContent = s;
        return div.innerHTML;
      }}

      function fetchThumbnails(ids) {{
        return new Promise(function(resolve, reject) {{
          var results = {{}};
          var remaining = ids.slice();
          function next() {{
            if (remaining.length === 0) {{ resolve(results); return; }}
            var batch = remaining.splice(0, BATCH);
            var url = USE_PROXY
              ? "/api/thumbnails?ids=" + encodeURIComponent(batch.join(','))
              : THUMB_API + "?assetIds=" + encodeURIComponent(batch.join(',')) + "&size=250x250&format=Png&isCircular=false";
            fetch(url).then(function(r) {{ return r.json(); }}).then(function(data) {{
              (data.data || []).forEach(function(item) {{
                if (item.state === 'Completed' && item.imageUrl)
                  results[String(item.targetId)] = item.imageUrl;
              }});
              setTimeout(next, USE_PROXY ? 0 : 300);
            }}).catch(function(err) {{ if (remaining.length === ids.length) reject(err); else setTimeout(next, 300); }});
          }}
          next();
        }});
      }}

      function fetchNames(ids) {{
        if (!USE_PROXY) return Promise.resolve({{}});
        return fetch("/api/names?ids=" + encodeURIComponent(ids.join(','))).then(function(r) {{ return r.json(); }}).catch(function() {{ return {{}}; }});
      }}

      window.loadGroupInSidePanel = function(idsJson) {{
        var ids = JSON.parse(idsJson);
        panel.classList.add('visible');
        bodyEl.innerHTML = '<p class="loading">Loading thumbnails and names...</p>';
        var thumbPromise = fetchThumbnails(ids);
        var namesPromise = fetchNames(ids);
        Promise.all([thumbPromise, namesPromise]).then(function(arr) {{
          var thumbUrls = arr[0];
          var names = arr[1] || {{}};
          var html = '<div class="cards">';
          ids.forEach(function(id) {{
            var url = CATALOG_URL + id;
            var thumb = thumbUrls[id];
            var name = names[id] || ('Asset ' + id);
            var img = thumb
              ? '<img src="' + esc(thumb) + '" alt="' + esc(name) + '" loading="lazy">'
              : '<span class="placeholder">[No thumb]</span>';
            var skeleton = USE_PROXY ? '<img class="skeleton-gif" src="/api/skeleton_gif?asset_id=' + esc(id) + '" alt="Skeleton" loading="lazy" onerror="this.style.display=\\'none\\'">' : '';
            html += '<div class="card"><a href="' + esc(url) + '" target="_blank" rel="noopener">' + img + '</a>' + skeleton + '<div class="meta"><span class="name">' + esc(name) + '</span><span class="id">' + esc(id) + '</span></div></div>';
          }});
          html += '</div>';
          html += '<p style="margin-top:0.75rem; color:#888; font-size:0.9rem;">Open in new tabs:</p><ul class="link-list" id="sideLinkList"></ul>';
          html += '<button type="button" id="openAllNewTabs" style="margin-top:0.5rem; padding:0.4rem 0.8rem; background:#0e639c; color:#fff; border:none; border-radius:4px; cursor:pointer;">Open all in new tabs</button>';
          bodyEl.innerHTML = html;
          var listEl = document.getElementById('sideLinkList');
          ids.forEach(function(id, i) {{
            var li = document.createElement('li');
            var a = document.createElement('a');
            a.href = CATALOG_URL + id;
            a.target = '_blank';
            a.rel = 'noopener';
            a.textContent = (names[id] || ('Asset ' + id)) + ' (' + id + ')';
            li.appendChild(a);
            listEl.appendChild(li);
          }});
          document.getElementById('openAllNewTabs').onclick = function() {{
            var delay = 0;
            ids.forEach(function(id) {{
              setTimeout(function() {{ window.open(CATALOG_URL + id, '_blank', 'noopener'); }}, delay);
              delay += 300;
            }});
          }};
        }}).catch(function() {{
          bodyEl.innerHTML = '<p class="loading">Failed to load. ' + (USE_PROXY ? 'Check server console.' : 'Run with <code>--lazy --serve</code> and open the page from the URL it prints (CORS blocks direct Roblox requests).') + '</p>';
        }});
      }};

      document.querySelectorAll('.load-in-panel').forEach(function(btn) {{
        btn.onclick = function() {{ window.loadGroupInSidePanel(btn.getAttribute('data-ids')); }};
      }});
      document.getElementById('sideFrameClose').onclick = function() {{ panel.classList.remove('visible'); }};
    }})();
  </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    # Default paths relative to repo root (parent of visualization/) so it works from any cwd
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_script_dir)
    _default_input = os.path.join(_repo_root, "dupes.csv")
    _default_output = os.path.join(_repo_root, "dupes_review.html")

    ap = argparse.ArgumentParser(description="Generate moderation review HTML from dupes.csv or groups.csv")
    ap.add_argument("--input", "-i", default=_default_input, help="Input CSV (dupes or groups format)")
    ap.add_argument("--output", "-o", default=None, help="Output HTML path (default: dupes_review.html next to input)")
    ap.add_argument("--no-fetch", action="store_true", help="Skip fetching thumbnails/names from Roblox; use placeholders")
    ap.add_argument("--no-names", action="store_true", help="Skip fetching names (faster); use 'Asset {id}'")
    ap.add_argument("--lazy", action="store_true", help="On-demand: show only group IDs; thumbnails/names load in side panel when you select a group")
    ap.add_argument("--serve", action="store_true", help="With --lazy: run a local server so thumbnails/names work (avoids CORS). Open the URL it prints.")
    ap.add_argument("--port", type=int, default=8765, help="Port for --serve (default: 8765)")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print API errors and non-Completed thumbnail states")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input) or ".", "dupes_review.html")

    groups = detect_and_load(args.input)
    if not groups:
        raise SystemExit("No groups found in CSV.")
    if args.lazy:
        use_proxy = args.serve
        render_html_lazy(groups, args.output, use_proxy=use_proxy)
        if args.serve:
            print(f"Wrote {args.output} ({len(groups)} groups).")
            serve_lazy(args.output, args.port)
        else:
            print(f"Wrote {args.output} ({len(groups)} groups). For thumbnails/names in the side panel, run with: --lazy --serve")
        return
    all_ids = []
    for g in groups:
        for aid in g:
            if aid and aid not in all_ids:
                all_ids.append(aid)
    if args.no_fetch:
        thumb_urls = {}
        asset_names = {}
    else:
        print("Fetching thumbnails from Roblox...")
        thumb_urls = fetch_thumbnails(all_ids, verbose=args.verbose)
        if args.no_names:
            asset_names = {aid: f"Asset {aid}" for aid in all_ids}
            print("Skipping names (--no-names).")
        else:
            print("Fetching names from Roblox (slow, rate-limited)...")
            asset_names = fetch_asset_names(all_ids, verbose=args.verbose)
        if args.verbose:
            print(f"  Got {len(thumb_urls)} thumbnails, {len(asset_names)} names")
    render_html(groups, args.output, thumb_urls, asset_names)
    print(f"Wrote {args.output} ({len(groups)} groups)")


if __name__ == "__main__":
    main()
