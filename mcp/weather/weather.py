import httpx
from mcp.server.fastmcp import FastMCP
from typing import Any

mcp = FastMCP("weather")

NEWS_API = "https://api.weather.gov"


async def make_api_request(url: str) -> None | dict[str, Any]:
    header = {
        'Accept': 'application/geo+json'
    }
    async with httpx.AsyncClient() as client:
        try:
            reponse = await client.get(url=url, headers=header, timeout=30.0)
            return reponse.json()
        except Exception as e:
            print(f"Error: {e}")
            return None
        
def format_alert(feature: dict) -> str:
    props = feature
    print(props)
    return props

@mcp.tool()
async def get_alerts(state: str) -> str:
    link = f'{NEWS_API}/alerts/active/area/{state}'
    data = await make_api_request(link)
    print(data)

    return data

@mcp.tool()
async def get_forecast(latitute: float, longitude: float) -> str:
    link = f'{NEWS_API}/points/{latitute}/{longitude}'
    data = await make_api_request(link)
    print(data)

    return data

def main():
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()