import time
import requests

API_URL = "https://jatinsheoran2412--argus-stream-a-api-argusapi-fastapi-app.modal.run"

def main():
    print(f"Triggering analysis on remote endpoint: {API_URL}")
    
    # We trigger with bypass_cache=True to verify Moondream2 and VideoMAE execution on CUDA.
    url = f"{API_URL}/samples/avenue-1/analyze"
    params = {"bypass_cache": "true", "roi_sector": "full"}
    
    print("Sending POST request...")
    res = requests.post(url, params=params)
    print(f"Response Status Code: {res.status_code}")
    
    if res.status_code != 202:
        print("Failed to queue analysis.")
        print(res.text)
        return
        
    data = res.json()
    job_id = data["job_id"]
    print(f"Job queued successfully. Job ID: {job_id}")
    
    print("Polling job status...")
    while True:
        status_url = f"{API_URL}/jobs/{job_id}"
        status_res = requests.get(status_url)
        if status_res.status_code != 200:
            print(f"Error checking job status: {status_res.status_code}")
            print(status_res.text)
            break
            
        job_data = status_res.json()
        status = job_data.get("status")
        progress = job_data.get("progress_pct")
        step = job_data.get("step")
        
        print(f"Status: {status} | Progress: {progress}% | Step: {step}")
        
        if status == "completed":
            print("\nAnalysis completed successfully!")
            result = job_data["result"]
            print(f"Cache Hit: {result['analysis']['cache_hit']}")
            print(f"Runtime: {result['analysis']['runtime_sec']:.2f}s")
            
            summary = result['analysis']['summary']
            print(f"Peak Score: {summary['peak_score']:.3f} at {summary['peak_time_sec']:.2f}s")
            print(f"VLM Caption: {summary['vlm_caption']}")
            print(f"Number of frames: {len(result['analysis']['frames'])}")
            
            # Print details of the first frame with rollout/boxes to make sure they are present
            first_frame = result['analysis']['frames'][0]
            print(f"Frame 0 keys: {list(first_frame.keys())}")
            img_url = first_frame.get('image_data_url', '')
            print(f"Frame 0 image URL starts with data URI: {img_url.startswith('data:image/jpeg;base64,')}")
            print(f"Frame 0 image URL length: {len(img_url)}")
            break
        elif status == "failed":
            print("\nAnalysis job failed!")
            print(f"Error: {job_data.get('error')}")
            break
            
        time.sleep(5)

if __name__ == "__main__":
    main()
