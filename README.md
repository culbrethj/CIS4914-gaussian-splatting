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

### Backend Setup and Run

This is needed for the current video upload implementation
Python 3 is required. Create and activate a virtual environment.
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

Install Backend Dependencies
```bash
pip install fastapi uvicorn python-multipart opencv-python
```

Run the server.
```bash
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`.

---

### Gaussian Splat Demo

Follow above instructions to create virtual environment ***Use Python 3.11** and add the following packages

```bash
source venv/bin/activate
pip install numpy pycolmap
```

Available datasets to sub in for ```banana```: south-building, truck, banana

Run SfM:
```bash
cd backend
python sfm.py datasets/banana/images datasets/banana/sparse
```

Run Gaussian Splatting:
```bash
./opensplat datasets/banana -o datasets/banana/splat.ply -n 2000
```

Or just run pipeline.py
```bash
python pipeline.py truck 5000
```

Note: I built ```./opensplat``` on Mac and it depends on opencv and pytorch, so you probably have to rebuild it on your machine. Follow instructions on this github and copy the produced binary [https://github.com/pierotofy/OpenSplat]()

Since opensplat is hopefully temporary until we gain access to HiPerGator, it may or may not make sense to use Docker to containerize this.
