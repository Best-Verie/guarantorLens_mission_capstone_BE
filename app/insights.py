"""Portfolio insights: watchlist, super-guarantors, default communities."""
from typing import List

from fastapi import APIRouter, Depends

from . import network_data, scoring
from .auth import get_current_user
from .models import User
from .schema import CommunityStat, SuperGuarantor, WatchlistItem

router = APIRouter(tags=["insights"])


@router.get("/watchlist", response_model=List[WatchlistItem])
def watchlist(user: User = Depends(get_current_user)):
    return network_data.watchlist(scoring.MEMBERS)


@router.get("/insights/super-guarantors", response_model=List[SuperGuarantor])
def super_guarantors(user: User = Depends(get_current_user)):
    return network_data.super_guarantors(scoring.MEMBERS)


@router.get("/insights/communities", response_model=List[CommunityStat])
def communities(user: User = Depends(get_current_user)):
    return network_data.communities(scoring.MEMBERS)
