# Moodify — Mood-based Songs / Movies / Series Recommender

This repository contains a mood-based media recommender application.

Components
- backend/         — Node.js + Express API (auth, media CRUD, recommendation proxy)
- frontend/        — React app (login, mood selector, recommendations feed)
- services/recommender/ — Python FastAPI microservice (ALS + embeddings + FAISS)
- postgres         — PostgreSQL database (via docker-compose)
- docker-compose.yml

Quick start (local, using docker-compose)
1. Copy `.env.example` to `.env` and set values:
   - DATABASE_URL, JWT_SECRET, REFRESH_TOKEN_SECRET, (optional) OPENAI_API_KEY
2. Start services:
   docker-compose up --build
3. Seed DB (one-time):
   cd backend && node prisma/seed.js   # or run provided SQL
4. Open the frontend at http://localhost:3000

Notes
- Auth: access tokens (short-lived) + refresh tokens (HTTP-only secure cookie).
- Recommender: POST /rebuild triggers model rebuild; GET /recommend queries the recommender microservice.
- For production: secure cookies, HTTPS, rotate secrets, and schedule nightly model retraining.
