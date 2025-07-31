from src.config.logging import LoggerMixin
from fastapi import APIRouter 
from fastapi.responses import JSONResponse

health_router = APIRouter() 

class HealthHandler(LoggerMixin):
    async def check(self):
        try:
            self.logger.info("Health check endpoint called.")
            return JSONResponse(
                {
                    "status" : "ok" 
                }, status_code = 200 
            )
        except Exception as e:
            self.logger.exception("Health check failed due to an error.")
            return JSONResponse(
                {
                    "status": "error",
                    "detail": str(e)
                }, status_code = 500 
            )

health_handler = HealthHandler() 

@health_router.get("/health") 
async def health():
    return await health_handler.check() 