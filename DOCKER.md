# Docker Command Reference

Quick reference for running and managing the app via Docker Compose. All
commands assume you're in the project root (where `docker-compose.yml`
lives), and that Docker Desktop is running.

## Start / stop

```
docker compose up -d       # start both services (api + dashboard), detached
docker compose down        # stop and remove containers; DATA IS KEPT (named volume persists)
```

Or just double-click **`run.bat`** / **`stop.bat`** — they wrap these with
readiness-checking and auto-opening the browser.

## Check status

```
docker compose ps          # what's running right now
docker compose logs        # all logs, both services
docker compose logs -f     # follow logs live (Ctrl+C to stop watching)
docker compose logs api    # just the API service
docker compose logs dashboard
```

## Deleting indexed data

**We don't currently have a way to remove a single document** — only
everything at once. To wipe all indexed documents and start fresh:

```
docker compose down -v     # -v also removes the named volume (all indexed data)
docker compose up -d       # start again with a clean, empty index
```

If you want your sample data back afterward:

```
docker compose exec api python scripts/seed.py
```

⚠️ **`-v` deletes the indexed copy of everything you've uploaded — including your own real documents, not just sample data.** It doesn't touch the original files on your computer, so anything you still have a copy of can be re-uploaded, but anything indexed only through the app is gone. Don't run this without knowing what's currently indexed (`docker compose ps` won't tell you that — use `curl http://127.0.0.1:8000/v1/documents` or check the dashboard's sidebar first).

## Rebuilding after code changes

If you edit the source code, the running containers won't pick it up
automatically — rebuild the image and recreate the containers:

```
docker compose up -d --build
```

## Restarting just one service

```
docker compose restart api
docker compose restart dashboard
```

## Running one-off commands inside a container

```
docker compose exec api python scripts/seed.py     # (re-)seed sample data
docker compose exec api bash                       # open a shell inside the api container
docker compose exec dashboard bash                  # same, for the dashboard container
```

`exec` only works while the container is already running (`docker compose up -d` first).
If you need a shell in a container that *isn't* running, use `run` instead:

```
docker compose run --rm api bash
```

## Checking disk usage / cleaning up unused images

```
docker system df           # how much space Docker images/volumes are using
docker image prune         # remove unused (dangling) images
```
