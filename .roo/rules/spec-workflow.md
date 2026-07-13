# Spec Workflow MCP Rules

These rules define exactly how the agent must load and use the **Spec Workflow MCP** tools.  
They ensure every feature request or specification follows the approved workflow sequence and approval process.

---

## 🧭 spec-workflow-guide
**Purpose:**  
Load essential spec workflow instructions to guide feature development from idea to implementation.

**When to Call:**  
- **Call this tool FIRST** whenever a user requests spec creation, feature development, or mentions *specifications*.  
- Must be loaded **before any other spec tools**.  

**Workflow Sequence:**  
`Requirements → Design → Tasks → Implementation`

**Rules:**  
- Always follow the workflow exactly to avoid errors.  
- Do not call other spec tools until this guide is loaded.  
- This tool establishes the core context for the entire spec process.

---

## 🧱 steering-guide
**Purpose:**  
Load guide for creating project steering documents.

**When to Call:**  
- Call **ONLY** when the user explicitly requests a steering document or asks about architecture documentation.  
- Not part of the standard spec workflow.

**Outputs:**  
- Provides templates and guidance for:
  - `product.md`
  - `tech.md`
  - `structure.md`

**Rules:**  
- Follow the provided steering workflow exactly to avoid errors.  
- Do not call this guide automatically—only by explicit request.

---

## 📊 spec-status
**Purpose:**  
Display the comprehensive specification progress overview.

**When to Call:**  
- When resuming work on a spec or checking overall completion status.  
- After running, the agent should review `tasks.md` to view progress markers:
  - `[ ]` pending  
  - `[-]` in progress  
  - `[x]` completed  

**Parameters:**  
- `projectPath*` → Absolute path to the project root.  
- `specName*` → Name of the specification.  

**Rules:**  
- Use this tool to summarize current spec progress before continuing work.  
- Do not modify spec content during status retrieval.

---

## ✅ approvals
**Purpose:**  
Manage approval requests through the dashboard interface.

**When to Call:**  
- After creating each document that requires user or stakeholder approval.  
- Use to request, check, or delete approvals as directed.

**Parameters:**  
| Parameter | Requirement | Description |
|------------|--------------|-------------|
| `action*` | required | `request`, `status`, or `delete` |
| `projectPath` | required for `request` | Absolute path to project root |
| `approvalId` | required for `status` and `delete` | ID of approval request |
| `title` | required for `request` | Short description of what needs approval |
| `filePath` | required for `request` | Relative path to file needing approval |
| `type` | required for `request` | `"document"` or `"action"` |
| `category` | required for `request` | `"spec"` or `"steering"` |
| `categoryName` | required for `request` | Name of the spec or `"steering"` |

**Rules:**  
- Provide **filePath only**—never include document content in the request.  
- Wait for the user or approver to complete the review before proceeding.  
- The dashboard reads files directly from disk.  
- Always log approvals by category (`spec` or `steering`) and maintain a clean state via `delete` after completion.

---

## 🔔 Approval Notifications (Ask Question Integration)

**Purpose:**  
Ensure the agent notifies the user when a specification has been submitted for approval and awaits review.

**Tool:**  
`ask-question`

**When to Call:**  
- Immediately **after** creating an approval request using the `approvals` tool (`action: request`).  
- Only trigger once per approval submission.

**Behavior:**
1. The agent must call the `ask-question` tool with the following message:
   > "A specification has been submitted for approval.  
   > Please respond with either **'I've reviewed the spec'** to approve it, or **'cancel'** to reject or withdraw it."

2. The agent then **waits for the user’s response** before continuing.

**Response Handling:**
- If the user replies **"I've reviewed the spec"**:  
  → The agent should mark the approval as acknowledged and continue the workflow (next step or implementation phase).

- If the user replies **"cancel"**:  
  → The agent must call the `approvals` tool with  
    `action: delete`  
    to remove the approval request, then stop the workflow.

**Rules:**
- Never continue development or implementation while awaiting user confirmation.  
- The agent must clearly log that the spec was reviewed before proceeding.  
- If no response is received after a defined wait period, the agent may send a single reminder but must not auto-approve.

**Example Workflow:**

    Generate spec document.

    Call approvals (action: request).

    Trigger ask-question with review prompt.

    Wait for user:

        If "I've reviewed the spec" → proceed by checking for approvals using the `approvals` tool.

        If "cancel" → call approvals (action: delete) → stop workflow.

## When working with a tasks.md file
  - Always work through tasks sequentially, marking items of as they are completed.
  - do not signal completion until all tasks are marked as complete.

---

### 🔒 Enforcement Notes
- These rules override any previous MCP usage guidance.  
- Agents must **load `spec-workflow-guide` before using any spec or approval tools.**  
- The `ask-question` tool must be used for every new approval submission.  
- Deviation from this sequence may cause invalid or incomplete workflows.  
- The workflow must end with a confirmed approval before closing a spec.

---