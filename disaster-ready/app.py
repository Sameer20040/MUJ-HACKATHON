import asyncio
import json
import threading
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from shapely import wkt as shapely_wkt

import sys
import importlib

import src.models
if not hasattr(src.models, 'TrackedAsset'):
    importlib.reload(src.models)

from src.database import get_db, get_engine, get_session_local, Base
from src.models import (
    FeedStatus, HazardType, PlanStressTest, ResponsePlan, Scenario, ScenarioEvent,
    ScenarioStatus, SensorReading, SensorSite, TrackedAsset, AssetPosition, OfficialAlert
)
from src.ingest.runner import ingest_runner
from src.services.seed import load_sensor_sites, load_sample_plans
from src.services.geo import region_of, haversine_km

# Ensure any newly-added tables (tracking, official alerts, etc.) exist.
# create_all is idempotent — it only creates missing tables, never alters existing ones.
Base.metadata.create_all(bind=get_engine())
from src.services.scenario_generator import generate_scenarios_from_live_data, expire_old_scenarios
from src.services.stress_test_engine import StressTestEngine

# ---------------------------------------------------------------------------
# Startup: seed + background live-data ingester
# ---------------------------------------------------------------------------
load_sensor_sites()
from src.services.seed import load_sample_plans
load_sample_plans()

if 'ingester_started' not in st.session_state:
    st.session_state.ingester_started = True

    def run_ingester(session_local):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ingest_runner.start())

    def run_tracking(session_local):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        from src.tracking.simulator import run_tracking_loop
        loop.run_until_complete(run_tracking_loop(interval=5))

    SessionLocal = get_session_local()
    ingester_thread = threading.Thread(target=run_ingester, args=(SessionLocal,), daemon=True)
    ingester_thread.start()
    st.session_state.ingester_thread = ingester_thread

    tracking_thread = threading.Thread(target=run_tracking, args=(SessionLocal,), daemon=True)
    tracking_thread.start()
    st.session_state.tracking_thread = tracking_thread

st.set_page_config(
    page_title="DisasterReady",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------
def db_session():
    gen = get_db()
    return next(gen), gen


SEV_COLORS = {"low": "#2e7d32", "medium": "#f9a825", "high": "#ef6c00", "extreme": "#b71c1c"}
HAZARD_COLORS = {
    'earthquake': 'red', 'flood': 'blue', 'wildfire': 'orange', 'cyclone': 'purple',
    'landslide': 'brown', 'volcanic': 'darkred', 'tsunami': 'darkblue', 'rainfall': 'lightblue',
    'temperature': 'lightred', 'humidity': 'lightgreen', 'wind_speed': 'beige',
    'wind_gust': 'beige', 'weather': 'gray',
}

st.sidebar.title("DisasterReady")

# Region selector
if 'region' not in st.session_state:
    st.session_state.region = "All"

region = st.sidebar.selectbox(
    "Region",
    ["All", "India", "United States"],
    index=["All", "India", "United States"].index(st.session_state.region)
)
st.session_state.region = region

page = st.sidebar.radio("Go to", ["Dashboard", "Map View", "Scenario Manager",
                                  "Plan Manager", "Stress Test Results", "Resource Optimizer", "Live Tracking"])


# ===========================================================================
# PAGE: Dashboard
# ===========================================================================
if page == "Dashboard":
    st.title("Dashboard")
    st.write("Live hazard monitoring and pre-emptive planning — all readings are real, live API data.")

    st.subheader("Data Ingestion Status")
    db, gen = db_session()
    try:
        feed_statuses = db.query(FeedStatus).all()
        if feed_statuses:
            cols = st.columns(len(feed_statuses))
            for i, status in enumerate(feed_statuses):
                icon = {"ok": "✅", "degraded": "⚠️", "error": "❌"}.get(status.status or "", "❔")
                mins = ""
                if status.last_update:
                    mins = f"{(datetime.utcnow() - status.last_update).total_seconds() / 60:.0f} min ago"
                with cols[i]:
                    st.metric(label=f"{icon} {status.feed_name}", value=status.status or "unknown",
                              delta=f"{mins} · {status.record_count or 0} readings", delta_color="off")
        else:
            st.info("No feed status yet — ingester will populate within a few minutes.")
    finally:
        try: next(gen)
        except StopIteration: pass

    # Quick actions
    st.subheader("Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌀 Generate Scenarios from Live Data", use_container_width=True):
            db, gen = db_session()
            try:
                created = generate_scenarios_from_live_data(db)
                expire_old_scenarios(db)
                st.success(f"Generated {len(created)} scenarios from real live readings.")
            except Exception as e:
                st.error(f"Scenario generation failed: {e}")
            finally:
                try: next(gen)
                except StopIteration: pass
    with col2:
        if st.button("🧪 Run All Stress Tests", use_container_width=True):
            db, gen = db_session()
            try:
                plan = db.query(ResponsePlan).filter(ResponsePlan.is_active == True).first()
                scenarios = db.query(Scenario).filter(Scenario.status != ScenarioStatus.expired).all()
                if not plan:
                    st.warning("No active plan found — create one in Plan Manager.")
                elif not scenarios:
                    st.warning("No scenarios yet — generate them from live data first.")
                else:
                    engine = StressTestEngine()
                    prog = st.progress(0.0)
                    for i, sc in enumerate(scenarios):
                        engine.run(db, plan, sc)
                        prog.progress((i + 1) / len(scenarios))
                    st.success(f"Stress-tested {plan.name} against {len(scenarios)} scenarios.")
            except Exception as e:
                st.error(f"Stress testing failed: {e}")
            finally:
                try: next(gen)
                except StopIteration: pass

    # Live sensor feed table
    st.subheader("Live Sensor Feed")
    db, gen = db_session()
    try:
        from src.services.geo import region_of
        sites = db.query(SensorSite).filter(SensorSite.is_active == True).all()
        # Filter by selected region
        if st.session_state.region != "All":
            sites = [s for s in sites if s.region == st.session_state.region or region_of(s.latitude, s.longitude) == st.session_state.region]
        if not sites:
            st.write("No active sensor sites found for this region.")
        else:
            rows = []
            for site in sites:
                latest = (db.query(SensorReading)
                          .filter(SensorReading.sensor_site_id == site.id)
                          .order_by(SensorReading.timestamp.desc()).first())
                htype = site.hazard_type.value if hasattr(site.hazard_type, 'value') else site.hazard_type
                if latest:
                    age_min = (datetime.utcnow() - latest.timestamp.replace(tzinfo=None)).total_seconds() / 60
                    fresh = "🟢" if age_min < 60 else ("🟡" if age_min < 24 * 60 else "🔴")
                    rows.append({
                        "Site": site.name, "Hazard": htype,
                        "Latest": f"{latest.value} {latest.unit}",
                        "Freshness": fresh, "Age (min)": round(age_min),
                        "Source": latest.source,
                        "Last Updated": latest.timestamp.strftime("%Y-%m-%d %H:%M"),
                    })
                else:
                    rows.append({"Site": site.name, "Hazard": htype, "Latest": "—", "Freshness": "⚪",
                                 "Age (min)": None, "Source": "", "Last Updated": "Never"})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            total = db.query(SensorReading).count()
            st.caption(f"📊 {total:,} real readings accumulated in the database from live public APIs. Showing region: {st.session_state.region}")
    finally:
        try: next(gen)
        except StopIteration: pass

    # Scenario counts by severity
    st.subheader("Scenario Counts by Severity")
    db, gen = db_session()
    try:
        sc_all = db.query(Scenario).filter(Scenario.status != ScenarioStatus.expired).all()
        if sc_all:
            sev_counts = {}
            for s in sc_all:
                sev = s.severity.value if hasattr(s.severity, 'value') else str(s.severity)
                sev_counts[sev] = sev_counts.get(sev, 0) + 1
            c1, c2, c3, c4 = st.columns(4)
            for col, sev in zip([c1, c2, c3, c4], ["low", "medium", "high", "extreme"]):
                with col:
                    st.metric(label=sev.title(), value=sev_counts.get(sev, 0))
        else:
            st.info("No active scenarios. Use 'Generate Scenarios from Live Data' above.")

        plans = db.query(ResponsePlan).filter(ResponsePlan.is_active == True).all()
        if plans:
            st.subheader("Plan Readiness")
            pcols = st.columns(len(plans))
            for col, p in zip(pcols, plans):
                with col:
                    score = p.stress_test_score
                    score_disp = f"{score:.0f}/100" if score is not None else "not tested"
                    st.metric(label=p.name, value=score_disp)
    finally:
        try: next(gen)
        except StopIteration: pass

    # Auto-refresh every 30 seconds to show live data
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = 0
    if time.time() - st.session_state.last_refresh > 30:
        st.session_state.last_refresh = time.time()
        st.rerun()


# ===========================================================================
# PAGE: Map View
# ===========================================================================
elif page == "Map View":
    st.title("Map View")
    st.write("Interactive map of live sensors, scenarios, evacuation routes, and resource positions.")

    try:
        from streamlit_folium import st_folium
        import folium
    except ImportError:
        st.error("streamlit-folium is not installed. Run: pip install streamlit-folium folium")
        st.stop()

    db, gen = db_session()
    try:
        # Filter sites by region
        sites = db.query(SensorSite).filter(SensorSite.is_active == True).all()
        if st.session_state.region != "All":
            sites = [s for s in sites if region_of(s.latitude, s.longitude) == st.session_state.region]

        lats = [s.latitude for s in sites if s.latitude is not None]
        lons = [s.longitude for s in sites if s.longitude is not None]

        # Determine map center based on region
        region_centers = {
            "India": [22.0, 79.0],
            "United States": [39.0, -98.0],
            "All": [20.0, 0.0]
        }
        center = region_centers.get(st.session_state.region, [20.0, 0.0])
        zoom_start = 5 if st.session_state.region == "All" else 7

        layers = st.multiselect("Layers", ["Live Sensors", "Scenario Areas", "Evacuation Routes", "Resource Depots",
                                   "Tracked Assets", "Official Alerts", "India State Boundaries", "Sensor Heatmap"],
                                default=["Live Sensors", "Scenario Areas", "Evacuation Routes", "Resource Depots"])
        m = folium.Map(location=center, zoom_start=zoom_start)

        if "Live Sensors" in layers:
            for site in sites:
                if site.latitude is None or site.longitude is None:
                    continue
                latest = (db.query(SensorReading)
                          .filter(SensorReading.sensor_site_id == site.id)
                          .order_by(SensorReading.timestamp.desc()).first())
                htype = site.hazard_type.value if hasattr(site.hazard_type, 'value') else site.hazard_type
                if latest:
                    popup = (f"<b>{site.name}</b><br/>Hazard: {htype}<br/>"
                             f"Value: {latest.value} {latest.unit}<br/>Source: {latest.source}<br/>"
                             f"Updated: {latest.timestamp.strftime('%Y-%m-%d %H:%M')}")
                else:
                    popup = f"<b>{site.name}</b><br/>Hazard: {htype}<br/>No readings yet"
                folium.Marker(
                    location=[site.latitude, site.longitude],
                    popup=folium.Popup(popup, max_width=300),
                    icon=folium.Icon(color=HAZARD_COLORS.get(htype, 'gray'), icon='info-sign'),
                ).add_to(m)

        if "Scenario Areas" in layers:
            active_scenarios = db.query(Scenario).filter(Scenario.status != ScenarioStatus.expired).all()
            for sc in active_scenarios:
                try:
                    poly = shapely_wkt.loads(sc.affected_area)
                    pts = [(y, x) for x, y in poly.exterior.coords]
                    sev = sc.severity.value if hasattr(sc.severity, 'value') else str(sc.severity)
                    folium.Polygon(
                        locations=pts,
                        color=SEV_COLORS.get(sev, 'gray'),
                        fill=True, fill_opacity=0.15, weight=2,
                        popup=folium.Popup(
                            f"<b>{sc.hazard_type.value} scenario</b><br/>Severity: {sev}<br/>"
                            f"Probability: {sc.probability:.0%}<br/>Horizon: {sc.time_horizon_hours}h",
                            max_width=250),
                    ).add_to(m)
                except Exception:
                    continue

        if "Evacuation Routes" in layers:
            plans = db.query(ResponsePlan).filter(ResponsePlan.is_active == True).all()
            for plan in plans:
                for route in plan.routes:
                    try:
                        line = shapely_wkt.loads(route.path)
                        pts = [(y, x) for x, y in line.coords]
                        color = 'green' if (route.capacity_per_hour or 0) >= 3500 else ('yellow' if (route.capacity_per_hour or 0) >= 2000 else 'red')
                        folium.PolyLine(
                            pts, color=color, weight=4,
                            popup=folium.Popup(f"<b>{route.route_name}</b><br/>Plan: {plan.name}<br/>"
                                               f"Capacity: {route.capacity_per_hour}/h", max_width=250),
                        ).add_to(m)
                    except Exception:
                        continue

        if "Resource Depots" in layers:
            plans = db.query(ResponsePlan).filter(ResponsePlan.is_active == True).all()
            if st.session_state.region != "All":
                plans = [p for p in plans if p.region_name == st.session_state.region]
            icons = {"medical_supplies": "plus", "food_water": "cutlery", "shelters": "home",
                     "rescue_teams": "users", "fuel": "truck", "communication": "signal"}
            for plan in plans:
                for res in plan.resources:
                    try:
                        pt = shapely_wkt.loads(res.location)
                        folium.Marker(
                            location=[pt.y, pt.x],
                            popup=folium.Popup(
                                f"<b>{res.resource_type.replace('_', ' ').title()}</b><br/>Plan: {plan.name}<br/>"
                                f"Qty: {res.quantity}<br/>Deploy: {res.deployment_time_hours or '?'}h", max_width=250),
                            icon=folium.Icon(color='darkgreen', icon=icons.get(res.resource_type, 'box'), prefix='glyphicon'),
                        ).add_to(m)
                    except Exception:
                        continue

        # India State Boundaries layer
        if "India State Boundaries" in layers and st.session_state.region in ["India", "All"]:
            try:
                import json
                from shapely.geometry import shape
                with open('seed_data/india_states.geojson', 'r') as f:
                    india_states = json.load(f)
                folium.GeoJson(
                    india_states,
                    name="India State Boundaries",
                    style_function=lambda x: {
                        'fillColor': 'transparent',
                        'color': '#333333',
                        'weight': 1.5,
                        'fillOpacity': 0.0
                    },
                    highlight_function=lambda x: {
                        'fillColor': '#ffff00',
                        'color': '#000000',
                        'weight': 3,
                        'fillOpacity': 0.3
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name', 'state_code'],
                        aliases=['State: ', 'Code: '],
                        style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
                    )
                ).add_to(m)
                # Add centroid markers for each state
                for feature in india_states['features']:
                    geom = shape(feature['geometry'])
                    if geom.is_valid:
                        centroid = geom.centroid
                        props = feature['properties']
                        state_name = props.get('name', 'Unknown')
                        state_code = props.get('state_code', '')
                        popup_text = f"<b>{state_name}</b>"
                        if state_code:
                            popup_text += f"<br/>Code: {state_code}"
                        folium.Marker(
                            location=[centroid.y, centroid.x],
                            popup=folium.Popup(popup_text, max_width=200),
                            icon=folium.Icon(color='darkblue', icon='map-marker', prefix='fa'),
                        ).add_to(m)
            except Exception as e:
                st.warning(f"Could not load India state boundaries: {e}")

        # Heatmap layer for sensor readings
        if "Sensor Heatmap" in layers:
            try:
                from folium.plugins import HeatMap
                heat_data = []
                for site in sites:
                    if site.latitude and site.longitude:
                        latest = (db.query(SensorReading)
                                  .filter(SensorReading.sensor_site_id == site.id)
                                  .order_by(SensorReading.timestamp.desc()).first())
                        if latest:
                            # Weight by hazard severity value
                            weight = float(latest.value) if latest.value else 1.0
                            heat_data.append([site.latitude, site.longitude, weight])
                if heat_data:
                    HeatMap(heat_data, radius=20, blur=15, max_zoom=10).add_to(m)
            except Exception as e:
                st.warning(f"Could not create heatmap: {e}")

        # New layers: Tracked Assets and Official Alerts
        if "Tracked Assets" in layers:
            assets = db.query(TrackedAsset).filter(TrackedAsset.is_active == True).all()
            if st.session_state.region != "All":
                assets = [a for a in assets if a.region == st.session_state.region]
            asset_colors = {
                "rescue_team": "red", "ambulance": "red", "supply_truck": "blue",
                "food_water": "green", "shelters": "orange", "fire_engine": "darkred",
                "evac_bus": "purple"
            }
            for asset in assets:
                if asset.latitude and asset.longitude:
                    color = asset_colors.get(asset.asset_type, "gray")
                    popup = f"""
                    <b>{asset.label}</b><br/>
                    Type: {asset.asset_type.replace('_', ' ').title()}<br/>
                    Status: {asset.status}<br/>
                    Speed: {asset.speed_kmh} km/h<br/>
                    Progress: {asset.progress:.0%}<br/>
                    <i>simulated movement</i>
                    """
                    folium.Marker(
                        location=[asset.latitude, asset.longitude],
                        popup=folium.Popup(popup, max_width=250),
                        icon=folium.Icon(color=color, icon="truck", prefix='fa')
                    ).add_to(m)

        if "Official Alerts" in layers:
            alerts = db.query(OfficialAlert).all()
            if st.session_state.region != "All":
                alerts = [a for a in alerts if a.region == st.session_state.region]
            for alert in alerts:
                if alert.geometry:
                    try:
                        geom = shapely_wkt.loads(alert.geometry)
                        severity_color = {"extreme": "red", "high": "orange", "medium": "yellow", "low": "green"}.get(alert.severity, "blue")
                        if geom.geom_type == "Point":
                            folium.CircleMarker(
                                location=[geom.y, geom.x],
                                radius=8,
                                popup=f"""
                                <b>{alert.event}</b><br/>
                                {alert.headline}<br/>
                                Severity: {alert.severity}<br/>
                                Area: {alert.area_desc}<br/>
                                <i>Source: GDACS</i>
                                """,
                                color=severity_color,
                                fill=True,
                                fill_opacity=0.7
                            ).add_to(m)
                        elif geom.geom_type in ["Polygon", "MultiPolygon"]:
                            centroid = geom.centroid
                            folium.CircleMarker(
                                location=[centroid.y, centroid.x],
                                radius=8,
                                popup=f"""
                                <b>{alert.event}</b><br/>
                                {alert.headline}<br/>
                                Severity: {alert.severity}<br/>
                                Area: {alert.area_desc}<br/>
                                <i>Source: GDACS</i>
                                """,
                                color=severity_color,
                                fill=True,
                                fill_opacity=0.7
                            ).add_to(m)
                    except Exception:
                        pass

        # Add layer controls
        folium.LayerControl().add_to(m)

        st_folium(m, width=None, height=550, use_container_width=True)
        st.caption("🟢 route ≥3500/h · 🟡 2000–3500/h · 🔴 <2000/h. Polygon color = scenario severity. Asset movement = SIMULATED.")
    finally:
        try: next(gen)
        except StopIteration: pass


# ===========================================================================
# PAGE: Scenario Manager
# ===========================================================================
elif page == "Scenario Manager":
    st.title("Scenario Manager")
    st.write("Scenarios derived from real live sensor trends and thresholds (Monte Carlo sampled peaks).")

    db, gen = db_session()
    try:
        if st.button("🌀 Generate Scenarios from Live Data"):
            created = generate_scenarios_from_live_data(db)
            st.success(f"Generated {len(created)} scenarios.")
            st.rerun()

        scenarios = db.query(Scenario).order_by(Scenario.created_at.desc()).all()
        if not scenarios:
            st.info("No scenarios yet. Generate them from live data above.")
            st.stop()

        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            hazard_filter = st.multiselect("Hazard", sorted({s.hazard_type.value for s in scenarios}))
        with f2:
            sev_filter = st.multiselect("Severity", ["low", "medium", "high", "extreme"])
        with f3:
            status_filter = st.multiselect("Status", ["simulating", "complete", "expired"], default=["complete", "simulating"])

        filtered = scenarios
        if hazard_filter:
            filtered = [s for s in filtered if s.hazard_type.value in hazard_filter]
        if sev_filter:
            filtered = [s for s in filtered if (s.severity.value if hasattr(s.severity, 'value') else str(s.severity)) in sev_filter]
        if status_filter:
            filtered = [s for s in filtered if (s.status.value if hasattr(s.status, 'value') else str(s.status)) in status_filter]

        rows = []
        for s in filtered:
            sev = s.severity.value if hasattr(s.severity, 'value') else str(s.severity)
            stat = s.status.value if hasattr(s.status, 'value') else str(s.status)
            rows.append({
                "ID": s.id, "Hazard": s.hazard_type.value, "Severity": sev.title(),
                "Probability": f"{s.probability:.0%}", "Horizon (h)": s.time_horizon_hours,
                "Status": stat, "Created": s.created_at.strftime("%Y-%m-%d %H:%M"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Detail view
        st.subheader("Scenario Detail")
        sc_id = st.selectbox("Select scenario", [f"#{s.id} — {s.hazard_type.value} / {s.severity.value} (p={s.probability:.0%})" for s in filtered])
        selected = filtered[[f"#{s.id}" for s in filtered].index(sc_id.split(" —")[0])]

        events = db.query(ScenarioEvent).filter(ScenarioEvent.scenario_id == selected.id).order_by(ScenarioEvent.timestamp_offset_hours).all()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Hazard:** {selected.hazard_type.value}  ")
            st.markdown(f"**Severity:** {selected.severity.value}  ")
            st.markdown(f"**Probability:** {selected.probability:.0%}  ")
        with c2:
            st.markdown(f"**Horizon:** {selected.time_horizon_hours}h  ")
            st.markdown(f"**Events:** {len(events)}  ")
            st.markdown(f"**Status:** {selected.status.value if hasattr(selected.status, 'value') else selected.status}")

        if events:
            st.subheader("Event Timeline")
            ev_df = pd.DataFrame([{
                "Offset (h)": e.timestamp_offset_hours,
                "Type": e.event_type,
                "Magnitude": e.magnitude,
                "Description": e.description,
            } for e in events])
            fig = px.timeline(
                ev_df, x_start=pd.to_datetime("2020-01-01") + pd.to_timedelta(ev_df["Offset (h)"], unit="h"),
                x_end=pd.to_datetime("2020-01-01") + pd.to_timedelta(ev_df["Offset (h)"] + 0.5, unit="h"),
                y=[f"#{selected.id}"] * len(ev_df), color="Type", hover_data=["Magnitude", "Description"],
            )
            fig.update_xaxes(title="Hours from scenario start")
            fig.update_layout(showlegend=True, height=240, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(ev_df, use_container_width=True, hide_index=True)

        # Stress-test this scenario against a plan
        st.subheader("Stress Test Against Plan")
        plans = db.query(ResponsePlan).filter(ResponsePlan.is_active == True).all()
        if plans and events:
            plan_name = st.selectbox("Plan", [p.name for p in plans])
            plan = next(p for p in plans if p.name == plan_name)
            if st.button("▶️ Run Stress Test"):
                engine = StressTestEngine()
                with st.spinner("Simulating multi-agent response..."):
                    test = engine.run(db, plan, selected)
                st.success(f"Overall score: **{test.overall_score}/100**")
                m1, m2, m3 = st.columns(3)
                with m1: st.metric("Evacuation Success", f"{test.evacuations_success_rate:.0%}")
                with m2: st.metric("Resource Coverage", f"{test.resource_coverage:.0%}")
                with m3: st.metric("Time to Response", f"{test.time_to_adequate_response_hours}h")
        elif not plans:
            st.info("No active plans. Create one in Plan Manager first.")
    finally:
        try: next(gen)
        except StopIteration: pass


# ===========================================================================
# PAGE: Plan Manager
# ===========================================================================
elif page == "Plan Manager":
    st.title("Plan Manager")
    db, gen = db_session()
    try:
        plans = db.query(ResponsePlan).all()
        for p in plans:
            score = f"{p.stress_test_score:.0f}/100" if p.stress_test_score is not None else "not tested"
            with st.expander(f"📋 {p.name} — v{p.version} ({p.hazard_type.value}) · Score: {score}"):
                c1, c2, c3 = st.columns(3)
                with c1: st.markdown(f"**Region:** {p.region_name}")
                with c2: st.markdown(f"**Hazard:** {p.hazard_type.value}")
                with c3: st.markdown(f"**Active:** {'Yes' if p.is_active else 'No'}")

                if p.plan_data:
                    try:
                        pdata = json.loads(p.plan_data)
                        if "trigger_thresholds" in pdata:
                            st.markdown("**Trigger Thresholds:**")
                            for k, v in pdata["trigger_thresholds"].items():
                                st.markdown(f"- `{k}`: {v}")
                    except Exception:
                        pass

                resources = p.resources
                routes = p.routes
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown(f"**Resources ({len(resources)})**")
                    for res in resources:
                        st.markdown(f"- {res.resource_type.replace('_',' ').title()} ×{res.quantity} @ {res.location}")
                with r2:
                    st.markdown(f"**Routes ({len(routes)})**")
                    for rt in routes:
                        st.markdown(f"- {rt.route_name} ({rt.capacity_per_hour or '?'}/h)")

                # Latest stress test result + revision
                last_test = (db.query(PlanStressTest).filter(PlanStressTest.plan_id == p.id)
                             .order_by(PlanStressTest.created_at.desc()).first())
                if last_test:
                    st.markdown(f"**Last stress test:** score {last_test.overall_score}/100 — "
                                f"evac {last_test.evacuations_success_rate:.0%}, coverage {last_test.resource_coverage:.0%}")
                    if last_test.recommended_revision:
                        rev = json.loads(last_test.recommended_revision)
                        n_rev = sum(len(v) for v in rev.values())
                        st.markdown(f"**Recommended revisions available:** {n_rev} (see Stress Test Results page)")

                if st.button("🧪 Run Stress Test (vs top scenarios)", key=f"test_{p.id}"):
                    scenarios = db.query(Scenario).filter(Scenario.status != ScenarioStatus.expired)\
                        .order_by(Scenario.probability.desc()).limit(5).all()
                    if not scenarios:
                        st.warning("No active scenarios — generate them from live data first.")
                    else:
                        engine = StressTestEngine()
                        prog = st.progress(0.0)
                        for i, sc in enumerate(scenarios):
                            engine.run(db, p, sc)
                            prog.progress((i + 1) / len(scenarios))
                        st.success(f"Tested against {len(scenarios)} scenarios. Refresh to see updated score.")
                        st.rerun()
    finally:
        try: next(gen)
        except StopIteration: pass

    st.subheader("Create New Plan")
    with st.form("new_plan_form"):
        name = st.text_input("Plan Name")
        hazard = st.selectbox("Primary Hazard", [h.value for h in HazardType])
        region = st.text_input("Region Name")
        submitted = st.form_submit_button("Create Plan")
        if submitted and name and region:
            db, gen = db_session()
            try:
                plan = ResponsePlan(
                    name=name,
                    hazard_type=HazardType(hazard),
                    region_name=region,
                    plan_data=json.dumps({"covered_cascades": [hazard], "trigger_thresholds": {}}),
                )
                db.add(plan)
                db.commit()
                st.success(f"Plan '{name}' created. Add routes/resources in a future revision.")
            except Exception as e:
                st.error(f"Failed: {e}")
                db.rollback()
            finally:
                try: next(gen)
                except StopIteration: pass


# ===========================================================================
# PAGE: Stress Test Results
# ===========================================================================
elif page == "Stress Test Results":
    st.title("Stress Test Results")
    db, gen = db_session()
    try:
        tests = db.query(PlanStressTest).order_by(PlanStressTest.created_at.desc()).all()
        if not tests:
            st.info("No stress tests yet. Run one from the Dashboard, Scenario Manager, or Plan Manager.")
            st.stop()

        # Heatmap plan × scenario
        st.subheader("Score Heatmap: Plan × Scenario")
        plans = {p.id: p.name for p in db.query(ResponsePlan).all()}
        scenarios = {s.id: f"#{s.id} {s.hazard_type.value}/{s.severity.value}" for s in db.query(Scenario).all()}
        heat_rows = []
        for t in tests:
            if t.plan_id in plans and t.scenario_id in scenarios:
                heat_rows.append({"Plan": plans[t.plan_id], "Scenario": scenarios[t.scenario_id], "Score": t.overall_score})
        if heat_rows:
            heat_df = pd.DataFrame(heat_rows).drop_duplicates(subset=["Plan", "Scenario"], keep="last")
            try:
                pivot = heat_df.pivot(index="Plan", columns="Scenario", values="Score")
                fig = px.imshow(pivot, text_auto=".0f", color_continuous_scale="RdYlGn", aspect="auto",
                                labels=dict(color="Score"), zmin=0, zmax=100)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.dataframe(heat_df, use_container_width=True, hide_index=True)

        st.subheader("Test History")
        for t in tests[:10]:
            plan_name = plans.get(t.plan_id, f"Plan {t.plan_id}")
            sc_name = scenarios.get(t.scenario_id, f"Scenario {t.scenario_id}")
            with st.expander(f"{'🟢' if (t.overall_score or 0) >= 60 else '🔴'} {plan_name} vs {sc_name} — {t.overall_score}/100"):
                m1, m2, m3 = st.columns(3)
                with m1: st.metric("Evacuation Success", f"{(t.evacuations_success_rate or 0):.0%}")
                with m2: st.metric("Resource Coverage", f"{(t.resource_coverage or 0):.0%}")
                with m3: st.metric("Time to Response", f"{t.time_to_adequate_response_hours or '?'}h")

                if t.bottlenecks:
                    st.markdown("**Bottlenecks Detected:**")
                    for b in json.loads(t.bottlenecks):
                        st.markdown(f"- ⚠️ {b.get('issue', 'unknown')}")
                if t.cascading_failures_detected:
                    casc = json.loads(t.cascading_failures_detected)
                    if casc:
                        st.markdown("**Cascading Failures (uncovered):**")
                        for c in casc:
                            st.markdown(f"- ⛓️ {c.get('description', '')} (+{c.get('offset_hours', '?')}h)")
                if t.recommended_revision:
                    rev = json.loads(t.recommended_revision)
                    st.markdown("**Actionable Revisions:**")
                    for r in rev.get("routes", []):
                        st.markdown(f"- 🛣️ `{r['action']}`: {r.get('suggestion', r.get('reason', ''))}")
                    for r in rev.get("resources", []):
                        if r["action"] == "relocate":
                            st.markdown(f"- 📦 Move depot `{r['resource_id']}` → `{r['to']}` ({r.get('reason','')}) — {r.get('benefit','')}")
                        else:
                            st.markdown(f"- 📦 Pre-position new {r.get('resource_type','')} cache at `{r.get('suggested_location','')}` — {r.get('reason','')}")
                    for r in rev.get("thresholds", []):
                        st.markdown(f"- ⏰ Add cascade trigger: {r.get('trigger','')} at +{r.get('at_offset_hours','?')}h")
    finally:
        try: next(gen)
        except StopIteration: pass


# ===========================================================================
# PAGE: Live Tracking
# ===========================================================================
elif page == "Live Tracking":
    st.title("Live Tracking")

    # Check for required modules
    try:
        from streamlit_geolocation import streamlit_geolocation
        import streamlit.components.v1 as components
        GEOLOCATION_AVAILABLE = True
    except ImportError:
        GEOLOCATION_AVAILABLE = False
        st.warning("streamlit-geolocation not installed. Install for live location sharing.")

    # Auto-refresh controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Live updating every 5 seconds")
    with col2:
        if st.button("🔄 Refresh Now"):
            st.rerun()

    # Get user location
    user_location = None
    if GEOLOCATION_AVAILABLE:
        location = streamlit_geolocation()
        if location and location.get('latitude') is not None:
            user_location = (location['latitude'], location['longitude'])
            st.success(f"📍 Your location: {user_location[0]:.4f}, {user_location[1]:.4f}")
        else:
            st.info("👆 Click the button above to share your location")
    else:
        # Fallback: manual coordinates for demo
        st.info("📍 Using demo location (Mumbai)")
        user_location = (19.0760, 72.8777)

    db, gen = db_session()
    try:
        # Evaluate user alerts if location available
        if user_location:
            lat, lon = user_location
            from src.services.alerting import evaluate_user_alerts
            alerts = evaluate_user_alerts(db, lat, lon)

            # Alert status banner
            if alerts:
                alert_severities = [a['severity'] for a in alerts]
                if 'extreme' in alert_severities:
                    st.error("🚨 EXTREME ALERT: Immediate action required!")
                elif 'high' in alert_severities:
                    st.warning("⚠️ HIGH ALERT: Take precautions")
                elif 'medium' in alert_severities:
                    st.info("ℹ️ MEDIUM ALERT: Stay informed")
                else:
                    st.success("✅ All clear - low severity alerts only")

                # Alert details
                with st.expander("Active Alerts", expanded=True):
                    for alert in alerts[:5]:  # Show top 5
                        severity_color = {
                            "extreme": "🔴",
                            "high": "🟠",
                            "medium": "🟡",
                            "low": "🟢"
                        }.get(alert['severity'], "⚪")
                        st.write(f"{severity_color} **{alert['severity'].title()}**: {alert['message']}")
                        if alert['distance_km'] > 0:
                            st.caption(f"   Distance: {alert['distance_km']:.1f} km | {alert['recommended_action']}")
            else:
                st.success("✅ No active alerts for your location")

        # Map visualization
        st.subheader("Live Situation Map")

        try:
            from streamlit_folium import st_folium
            import folium
            from shapely import wkt as shapely_wkt

            # Determine map center
            if user_location:
                map_center = [user_location[0], user_location[1]]
                zoom_start = 12
            else:
                # Default to region center
                region_centers = {
                    "India": [22.0, 79.0],
                    "United States": [39.0, -98.0],
                    "All": [20.0, 0.0]
                }
                map_center = region_centers.get(st.session_state.region, [20.0, 0.0])
                zoom_start = 5

            m = folium.Map(location=map_center, zoom_start=zoom_start)

            # Add user location marker
            if user_location:
                folium.Marker(
                    location=[user_location[0], user_location[1]],
                    popup="📍 Your Location",
                    icon=folium.Icon(color="green", icon="user", prefix='fa')
                ).add_to(m)

            # Add tracked assets
            assets = db.query(TrackedAsset).filter(TrackedAsset.is_active == True).all()
            # Filter by region if needed
            if st.session_state.region != "All":
                assets = [a for a in assets if a.region == st.session_state.region or
                         (a.latitude and a.longitude and region_of(a.latitude, a.longitude) == st.session_state.region)]

            # Asset type colors
            asset_colors = {
                "rescue_team": "red",
                "ambulance": "red",
                "supply_truck": "blue",
                "food_water": "green",
                "shelters": "orange",
                "fire_engine": "darkred",
                "evac_bus": "purple"
            }

            for asset in assets:
                if asset.latitude and asset.longitude:
                    color = asset_colors.get(asset.asset_type, "gray")
                    status_text = f" ({asset.status})" if asset.status != "staged" else ""
                    popup_text = f"""
                    <b>{asset.label}</b><br/>
                    Type: {asset.asset_type}{status_text}<br/>
                    Speed: {asset.speed_kmh} km/h<br/>
                    Progress: {asset.progress:.0%}<br/>
                    <i>simulated movement</i>
                    """

                    folium.Marker(
                        location=[asset.latitude, asset.longitude],
                        popup=folium.Popup(popup_text, max_width=250),
                        icon=folium.Icon(color=color, icon="truck", prefix='fa')
                    ).add_to(m)

                    # Show path if exists
                    if asset.path:
                        try:
                            path_geom = shapely_wkt.loads(asset.path)
                            if path_geom.geom_type == "LineString":
                                folium.PolyLine(
                                    locations=[[coord[1], coord[0]] for coord in path_geom.coords],
                                    color=color,
                                    weight=3,
                                    opacity=0.7
                                ).add_to(m)
                        except Exception:
                            pass

            # Add official alerts
            alerts_db = db.query(OfficialAlert).all()
            if st.session_state.region != "All":
                alerts_db = [a for a in alerts_db if a.region == st.session_state.region]

            for alert in alerts_db:
                if alert.geometry:
                    try:
                        geom = shapely_wkt.loads(alert.geometry)
                        if geom.geom_type == "Point":
                            folium.Marker(
                                location=[geom.y, geom.x],
                                popup=f"""
                                <b>{alert.event}</b><br/>
                                {alert.headline}<br/>
                                Severity: {alert.severity}<br/>
                                <i>Source: GDACS</i>
                                """,
                                icon=folium.Icon(
                                    color={"extreme": "red", "high": "orange", "medium": "yellow", "low": "green"}.get(alert.severity, "blue"),
                                    icon="exclamation-triangle",
                                    prefix='fa'
                                )
                            ).add_to(m)
                        elif geom.geom_type in ["Polygon", "MultiPolygon"]:
                            # For polygons, add a centroid marker
                            centroid = geom.centroid
                            folium.CircleMarker(
                                location=[centroid.y, centroid.x],
                                radius=5,
                                popup=f"""
                                <b>{alert.event}</b><br/>
                                {alert.headline}<br/>
                                Severity: {alert.severity}<br/>
                                Area: {alert.area_desc}
                                """,
                                color={"extreme": "red", "high": "orange", "medium": "yellow", "low": "green"}.get(alert.severity, "blue"),
                                fill=True,
                                fill_opacity=0.6
                            ).add_to(m)
                    except Exception:
                        pass

            # Add layer controls
            folium.LayerControl().add_to(m)

            # Display map
            st_folium(m, width=None, height=500)

        except ImportError:
            st.error("streamlit-folium not installed. Run: pip install streamlit-folium folium")

        # Asset roster table
        st.subheader("Asset Roster")
        if assets:
            asset_data = []
            for asset in assets:
                asset_data.append({
                    "Label": asset.label,
                    "Type": asset.asset_type.replace('_', ' ').title(),
                    "Status": asset.status.replace('_', ' ').title(),
                    "Speed (km/h)": asset.speed_kmh,
                    "Progress": f"{asset.progress:.0%}",
                    "Region": asset.region or region_of(asset.latitude, asset.longitude) if asset.latitude and asset.longitude else "Unknown",
                    "Last Updated": asset.updated_at.strftime("%H:%M:%S") if asset.updated_at else "Never"
                })

            if asset_data:
                import pandas as pd
                df = pd.DataFrame(asset_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Dispatch controls
                st.subheader("Dispatch Asset")
                with st.form("dispatch_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        dispatch_label = st.text_input("Asset Label", "Response Team")
                        dispatch_type = st.selectbox("Asset Type",
                                                   ["rescue_team", "ambulance", "supply_truck", "food_water", "shelters",
                                                    "fire_engine", "evac_bus"])
                    with col2:
                        dispatch_speed = st.number_input("Speed (km/h)", min_value=1, max_value=120, value=40)

                    # Get origin/target from user location or defaults
                    if user_location:
                        default_origin = f"POINT({user_location[1]} {user_location[0]})"
                        # Target 5km away in random direction for demo
                        import math
                        angle = math.radians(45)  # 45 degrees
                        distance_km = 5.0
                        earth_radius = 6371.0
                        lat_rad = math.radians(user_location[0])
                        lon_rad = math.radians(user_location[1])

                        target_lat = math.asin(math.sin(lat_rad) * math.cos(distance_km/earth_radius) +
                                              math.cos(lat_rad) * math.sin(distance_km/earth_radius) * math.cos(angle))
                        target_lon = lon_rad + math.atan2(math.sin(angle) * math.sin(distance_km/earth_radius) * math.cos(lat_rad),
                                                         math.cos(distance_km/earth_radius) - math.sin(lat_rad) * math.sin(target_lat))

                        target_lat = math.degrees(target_lat)
                        target_lon = math.degrees(target_lon)
                        default_target = f"POINT({target_lon} {target_lat})"
                    else:
                        default_origin = "POINT(72.8777 19.0760)"  # Mumbai
                        default_target = "POINT(72.9277 19.1260)"   # Nearby

                    dispatch_origin = st.text_input("Origin (WKT)", default_origin)
                    dispatch_target = st.text_input("Target (WKT)", default_target)

                    submitted = st.form_submit_button("🚀 Dispatch Asset")
                    if submitted:
                        from src.tracking.simulator import dispatch_asset
                        db_disp, gen_disp = db_session()
                        try:
                            asset_disp = asyncio.run(dispatch_asset(
                                db=db_disp,
                                label=dispatch_label,
                                asset_type=dispatch_type,
                                origin=dispatch_origin,
                                target=dispatch_target,
                                speed_kmh=dispatch_speed,
                                region=st.session_state.region if st.session_state.region != "All" else None
                            ))
                            if asset_disp:
                                st.success(f"Dispatched {dispatch_label} ({dispatch_type}) successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to dispatch asset. Check origin/target WKT format.")
                        except Exception as e:
                            st.error(f"Dispatch failed: {e}")
                        finally:
                            try: next(gen_disp)
                            except StopIteration: pass
            else:
                st.info("No assets match current filters")
        else:
            st.info("No tracked assets deployed yet")

        # Export buttons
        st.subheader("Export Situation Report")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📥 Export GeoJSON"):
                try:
                    from src.services.export import build_geojson
                    geojson_data = build_geojson(db, region=st.session_state.region if st.session_state.region != "All" else None)
                    st.download_button(
                        label="Download GeoJSON",
                        data=json.dumps(geojson_data, indent=2),
                        file_name=f"disasterready_{st.session_state.region.lower()}_situation.geojson",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Export failed: {e}")

        with col2:
            if st.button("📄 Export PDF"):
                try:
                    from src.services.export import build_situation_pdf
                    pdf_data = build_situation_pdf(db,
                                                 region=st.session_state.region if st.session_state.region != "All" else None,
                                                 user_location=user_location)
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name=f"disasterready_{st.session_state.region.lower()}_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Export failed: {e}")

        with col3:
            if st.button("📊 View Statistics"):
                stats = {
                    "Active Assets": len([a for a in assets if a.status != "staged"]),
                    "Total Assets": len(assets),
                    "Official Alerts": len(alerts_db) if 'alerts_db' in locals() else 0,
                    "User Alerts": len(alerts) if 'alerts' in locals() and user_location else 0
                }
                for key, value in stats.items():
                    st.metric(key, value)

    finally:
        try: next(gen)
        except StopIteration: pass

# ===========================================================================
# PAGE: Resource Optimizer
# ===========================================================================
elif page == "Resource Optimizer":
    st.title("Resource Optimizer")
    st.write("Gap analysis and pre-positioning recommendations derived from real scenario stress tests.")

    db, gen = db_session()
    try:
        tests = db.query(PlanStressTest).order_by(PlanStressTest.created_at.desc()).all()
        if not tests:
            st.info("Run stress tests first — the optimizer analyzes their results.")
            st.stop()

        # Aggregate recommendations across recent tests
        all_res_recs, gap_tests = [], [t for t in tests if (t.resource_coverage or 0) < 0.8]
        for t in gap_tests[:10]:
            if t.recommended_revision:
                rev = json.loads(t.recommended_revision)
                for r in rev.get("resources", []):
                    all_res_recs.append((t, r))

        st.subheader("Coverage Gaps")
        if gap_tests:
            gap_df = pd.DataFrame([{
                "Scenario": f"#{t.scenario_id}",
                "Resource Coverage": f"{(t.resource_coverage or 0):.0%}",
                "Score": t.overall_score,
            } for t in gap_tests])
            st.dataframe(gap_df, use_container_width=True, hide_index=True)
        else:
            st.success("All recent tests show adequate resource coverage (≥80%).")

        st.subheader("Pre-Positioning Recommendations")
        if all_res_recs:
            for t, r in all_res_recs[:10]:
                sc_name = f"Scenario #{t.scenario_id}"
                if r["action"] == "relocate":
                    st.markdown(f"- **Move** `{r.get('from','')}` → `{r.get('to','')}` — {r.get('reason','')} ({r.get('benefit','')}) · *{sc_name}*")
                else:
                    st.markdown(f"- **Pre-position** {r.get('resource_type','').replace('_',' ')} cache at `{r.get('suggested_location','')}` — {r.get('reason','')} · *{sc_name}*")
        else:
            st.info("No pre-positioning recommendations — coverage looks adequate.")

        # Map of suggested pre-position locations
        try:
            from streamlit_folium import st_folium
            import folium
            pts = []
            for t, r in all_res_recs:
                loc = r.get("to") or r.get("suggested_location")
                if loc:
                    try:
                        p = shapely_wkt.loads(loc)
                        pts.append((p.y, p.x))
                    except Exception:
                        pass
            if pts:
                m = folium.Map(location=[sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)], zoom_start=5)
                for lat, lon in pts:
                    folium.Marker(location=[lat, lon], icon=folium.Icon(color='orange', icon='flag')).add_to(m)
                st.subheader("Suggested New Positions")
                st_folium(m, width=None, height=400, use_container_width=True)
        except ImportError:
            pass
    finally:
        try: next(gen)
        except StopIteration: pass


st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit · Live data: USGS · Open-Meteo")
