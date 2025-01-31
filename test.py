import requests
import json
from typing import Dict, Any
import subprocess

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.headers = {'Content-Type': 'application/json'}

    def test_with_requests(self, review: str) -> Dict[str, Any]:
        """Test using Python requests library"""
        print("\n=== Testing with Python requests ===")
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                headers=self.headers,
                json={"review": review}
            )
            response.raise_for_status()
            result = response.json()
            print(f"Status Code: {response.status_code}")
            print("Response:")
            print(json.dumps(result, indent=2))
            return result
        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")
            return {}

    def test_with_curl(self, review: str) -> None:
        """Test using curl command"""
        print("\n=== Testing with curl ===")
        curl_command = f"""
        curl -X POST "{self.base_url}/predict" \\
             -H "Content-Type: application/json" \\
             -d '{{"review": "{review}"}}'
        """
        print("Curl command:")
        print(curl_command)
        
        try:
            result = subprocess.run(
                ['curl', '-X', 'POST', 
                 f"{self.base_url}/predict",
                 '-H', 'Content-Type: application/json',
                 '-d', json.dumps({"review": review})],
                capture_output=True,
                text=True
            )
            print("\nResponse:")
            print(json.dumps(json.loads(result.stdout), indent=2))
        except subprocess.SubprocessError as e:
            print(f"Error executing curl command: {str(e)}")

    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the health check endpoint"""
        print("\n=== Testing Health Endpoint ===")
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            result = response.json()
            print("Health Check Response:")
            print(json.dumps(result, indent=2))
            return result
        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")
            return {}

def main():
    tester = APITester()
    
    test_reviews = [
        "This movie was fantastic! The acting was great and I loved every moment.",
        "Terrible waste of time. The plot made no sense and the acting was horrible.",
        "It was an okay movie, but I've seen better."
    ]
    
    tester.test_health_endpoint()
    
    for review in test_reviews:
        print("\n" + "="*50)
        print(f"Testing review: {review}")
        print("="*50)
        
        tester.test_with_requests(review)
        
        tester.test_with_curl(review)
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()