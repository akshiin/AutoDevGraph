from graph import build_graph

USER_TASK = """
**Role**: You are a Senior System Architect specializing in FastAPI and Clean Architecture.
**Goal**: Transform the provided "Portfolio Platform" project description into a structured technical specification for a Developer Agent.

**Instructions**:
Analyze the project description and output a JSON or Markdown technical specification containing the following sections:

1. **Database Schema**:
- Define a 'User' (Author) model with all attributes (First Name, Last Name, Username, Profession, Birthdate, Email, Password Hash).
- Define a 'SocialLink' model (One-to-Many with User).
- Define a 'Project' model with attributes for Description, Cover S3 URL, and Visibility toggles.
- Define a 'Slide' model (One-to-Many with Project) with an 'order_index' and S3 image paths.

2. **Validation Rules (Pydantic)**:
- Create strict regex patterns for Name/Username (no spaces, no brackets, no punctuation).
- Implement the "12+ Age" logic (Birthdate validation).
- Implement Email mask validation.

3. **API Endpoints (FastAPI)**:
- `POST /auth/register`: Handle registration, password generation, and mock email trigger.
- `POST /auth/login`: JWT-based authentication.
- `GET/PUT /user/profile`: Personal data and Social Links management.
- `GET/POST/PUT /project`: Handle the single-project limitation (Max 1 project per user).
- `POST /project/slides`: Logic for max 12 slides and ordering.
- `GET /view/{protected_link}`: Public-facing view with visibility logic.

4. **Business Logic & Constraints**:
- Random 8-character password generation logic.
- S3 Integration: Define logic for private S3 storage for images.
- Project logic: Enforce "Step-by-step" tab progression (Description -> Content -> Publication).

5. **Security**:
- Password hashing (passlib/bcrypt).
- Protected routes using OAuth2 (JWT).

**Output Format**:
Provide a clear "Contract" that a developer can use to create models.py, schemas.py, and main.py. Do not write the code yourself; write the detailed architectural instructions.
"""

if __name__ == "__main__":
    app = build_graph()
    for output in app.stream({"task": USER_TASK}):
        print(output)
