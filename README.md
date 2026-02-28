# CIS4914 Project: End-to-end Gaussian Splatting Implementation

### Due Dates
- ~~**Feb 01: Project Proposal**~~
- ~~Feb 06: Week 4 reports~~
- Feb 13: Week 5 reports
- **Feb 15: Presentation 1 video**
- Feb 20: Week 6 reports
- Feb 27: Week 7 reports
- Mar 06: Week 8 reports
- Mar 13: Week 9 reports
- **Mar 13: Presentation 2 video**
- Mar 27: Week 11 reports
- Apr 03: Week 12 reports
- Apr 10: Week 13 reports
- **Apr 14: Senior Showcase**
- **Apr 21: Final Presentation video**

---

### Work Plan
- Milestone and Presentation 1: Week 4-5
    - Milestone 1: Video ingestion and SfM point cloud generation
        - Week 4 Tasks
            - Frame splicing from video
            - Spin up frontend directory
            - Run SfM on a sample dataset
            - Move this type of task list to Github Projects
        - Week 5 Tasks
            - Run frame splicing + SfM on a video we take
- Milestone and Presentation 2: Week 6-9
- Milestone and Presentation 3: Week 11-13

---

### Frontend Setup and Run

NodeJs is required (v20+). Check your Node version.

```bash
node -v
```

Install frontend dependencies and run.

```bash
cd frontend
npm install
```

Run the app.

```bash
npm run dev
```
Visit `http://localhost:5173` in browser.

--- 

### Gaussian Splat Demo

Run banana dataset:
```bash
cd backend
./opensplat datasets/banana -n 3000
```
