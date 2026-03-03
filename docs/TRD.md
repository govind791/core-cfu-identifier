# Technical Requirements Document (TRD)

## CFU Detection & Counting System for TFA Agar Plates

---

## 1. Document Control

| Item              | Value                                    |
| ----------------- | ---------------------------------------- |
| Project Name      | CFU Detection & Counting Service         |
| Client            | Core QC Labs                             |
| Backend Framework | FastAPI (Python)                         |
| Architecture      | Microservices, Async Job Processing      |
| Storage           | S3-compatible (MinIO local, AWS S3 prod) |
| Database          | PostgreSQL                               |
| Queue             | Redis + Worker (Celery / Arq / RQ)       |
| Version           | v1.0                                     |
| Status            | Final for Backend Development            |

---

## 2. Objective

Build a **robust, auditable FastAPI backend** that:

* Accepts **TFA agar plate images**
* Runs **asynchronous CFU detection & counting**
* Produces:

  * CFU count
  * Individual colony detections
  * Quality and confidence metrics
  * **Annotated image output when colonies are detected**
* Supports **multi-tenant lab operations**
* Is **traceable, reproducible, and review-friendly**

This system must be suitable for **regulated lab workflows**, not consumer demos.

---

## 3. Scope

### In Scope

* Image upload and validation
* Asynchronous processing pipeline
* Colony detection integration (ML-agnostic wrapper)
* Count calculation
* Quality control (QC) metrics
* Annotated image generation (conditional)
* Secure storage and retrieval
* API contracts and audit logging

### Out of Scope (Explicit)

* Manual review UI
* Colony species classification
* Regulatory submission logic (USP reporting)
* Model training (only inference integration)

---

## 4. High-Level Architecture

### Components

1. **API Service (FastAPI)**

   * Authentication
   * Request validation
   * Job orchestration
2. **Worker Service**

   * Image preprocessing
   * Model inference
   * Post-processing
   * Annotation rendering
3. **Redis**

   * Job queue
4. **PostgreSQL**

   * Metadata, results, audit logs
5. **Object Storage**

   * Raw images
   * Annotated images
   * Masks/debug artifacts

---

## 5. User & System Actors

* **Client System** (LIMS / Portal / API consumer)
* **Backend API**
* **Processing Worker**
* **Object Storage**
* **Database**

---

## 6. Input Requirements

### 6.1 Mandatory Inputs

| Field          | Type         | Notes                            |
| -------------- | ------------ | -------------------------------- |
| image          | file         | JPG / PNG only                   |
| sample_id      | string       | External reference               |
| plate_type     | enum         | `TFA_90MM`, `TFA_100MM`          |
| capture_method | enum         | `PHONE`, `SCANNER`, `CAMERA_RIG` |
| captured_at    | ISO datetime |                                  |
| client_id      | string       | Tenant isolation                 |

### 6.2 Optional (Strongly Recommended)

| Field            | Purpose        |
| ---------------- | -------------- |
| operator_id      | Traceability   |
| facility_id      | Audit          |
| dilution         | Interpretation |
| incubation_hours | Context        |
| lighting_type    | QC analysis    |

---

## 7. API Specifications

### 7.1 Create Job

`POST /v1/plates/jobs`

**Request**

* Multipart form data
* Metadata + image

**Response**

```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

---

### 7.2 Get Job Status

`GET /v1/plates/jobs/{job_id}`

```json
{
  "job_id": "uuid",
  "status": "queued | running | succeeded | failed",
  "progress": 0.0
}
```

---

### 7.3 Get Job Result

`GET /v1/plates/jobs/{job_id}/result`

(See Result Contract below)

---

### 7.4 Batch Upload

`POST /v1/plates/jobs/batch`

* Accepts multiple images
* Returns list of job IDs

---

## 8. Processing Pipeline (Worker)

### 8.1 Image Validation

* Minimum resolution check
* File integrity
* Supported formats only

### 8.2 Pre-Processing

* Auto-orientation
* Plate detection (circle/ellipse)
* Crop to plate ROI
* Lighting normalization
* Glare & blur scoring

### 8.3 Colony Detection (Model-Agnostic)

* Model inference via wrapper
* Output standardized to:

  * position
  * size
  * confidence score

### 8.4 Post-Processing

* Remove detections outside plate
* Non-max suppression
* Artifact filtering (bubbles, scratches)
* Cluster handling (basic watershed if applicable)

---

## 9. CFU Count Rules

| Scenario            | CFU Count |
| ------------------- | --------- |
| No colonies         | `0`       |
| Detectable colonies | Integer   |
| Overgrowth / TNTC   | `null`    |
| Invalid image       | `null`    |

**Counts are derived only from filtered detections.**

---

## 10. Annotated Image Rules (MANDATORY)

### Core Rule (Final, Non-Negotiable)

> **Annotated image MUST be generated if and only if `detections.length > 0`.**

### Explicit Behavior

| Condition              | Annotated Image   |
| ---------------------- | ----------------- |
| detections.length > 0  | MUST generate     |
| detections.length = 0  | MUST NOT generate |
| cfu_count_total = 0    | No annotation     |
| cfu_count_total = null | No annotation     |

### Annotation Contents

* Plate boundary (optional)
* Colony markers (circle or bbox)
* Colony index
* CFU total label

### Storage

```
/client_id/yyyy/mm/dd/job_id/annotated.png
```

---

## 11. Result Contract (Final)

```json
{
  "cfu_count_total": 42,
  "detections": [
    {
      "x": 0.45,
      "y": 0.51,
      "radius_px": 8,
      "score": 0.87
    }
  ],
  "quality": {
    "plate_found": true,
    "focus_score": 0.81,
    "glare_score": 0.12,
    "overgrowth_detected": false
  },
  "confidence": {
    "overall_score": 0.78,
    "needs_review": false,
    "reason_codes": []
  },
  "artifacts": {
    "annotated_image_url": "signed-url-or-null"
  },
  "model_metadata": {
    "model_name": "cfu-detector",
    "model_version": "v1.2.3",
    "pipeline_hash": "sha256"
  }
}
```

---

## 12. Quality & Review Rules

### needs_review = true if:

* focus_score < threshold
* glare_score > threshold
* overgrowth_detected = true
* plate_not_found = true
* CFU exceeds defined limit (e.g., >300)

---

## 13. Database Schema (Minimum)

### plate_jobs

* id
* client_id
* status
* created_at
* completed_at

### plate_images

* job_id
* image_url
* metadata (JSONB)

### plate_results

* job_id
* cfu_count_total
* detections (JSONB)
* quality (JSONB)
* confidence (JSONB)
* artifacts (JSONB)
* model_metadata (JSONB)

### audit_logs

* job_id
* action
* timestamp
* actor

---

## 14. Security & Multi-Tenancy

* API Key or JWT per client
* Client-scoped data access
* Signed URLs with expiry
* Full audit trail

---

## 15. Observability

* Structured logs (job_id, client_id)
* Metrics:

  * processing time
  * failure rate
  * review rate
* Model drift monitoring hooks

---

## 16. Deployment Requirements

* Dockerized services
* Docker-compose for local dev
* Separate API and worker containers
* Config via environment variables
* GPU optional but supported

---

## 17. Deliverables

Backend developer must deliver:

1. FastAPI service
2. Worker service
3. Queue integration
4. Storage integration
5. DB migrations
6. OpenAPI documentation
7. Test suite (API + worker)
8. Sample annotated output

---

## 18. Key Design Constraints (Read This Twice)

* **No inline ML in API threads**
* **No annotated images for zero counts**
* **Every result must be reproducible**
* **Every count must be explainable**
* **Failures are valid outcomes**