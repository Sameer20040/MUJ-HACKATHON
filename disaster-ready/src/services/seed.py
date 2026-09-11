import json
import os
from src.database import get_db
from src.models import SensorSite, ResponsePlan, ResourcePosition, EvacuationRoute, HazardType


def _seed_file(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'seed_data', filename))


def load_sensor_sites():
    """Load sensor sites from seed_data/sensor_sites.json if the table is empty."""
    # Get a database session
    db_gen = get_db()
    db = next(db_gen)
    try:
        # Check if we already have sensor sites
        count = db.query(SensorSite).count()
        if count > 0:
            print(f"Sensor sites already populated with {count} records. Skipping seed.")
            return

        seed_file = _seed_file('sensor_sites.json')
        print(f"Looking for seed file at: {seed_file}")
        if not os.path.exists(seed_file):
            print(f"Seed file not found: {seed_file}")
            return

        with open(seed_file, 'r') as f:
            sites_data = json.load(f)

        # Insert each site
        for site_data in sites_data:
            site = SensorSite(
                name=site_data['name'],
                hazard_type=site_data['hazard_type'],
                location=site_data['location'],  # This is WKT string
                upstream_url_template=site_data.get('upstream_url_template'),
                source=site_data['source'],
                is_active=site_data.get('is_active', True),
                site_code=site_data.get('site_code')
            )
            db.add(site)

        db.commit()
        print(f"Successfully seeded {len(sites_data)} sensor sites.")

    except Exception as e:
        print(f"Error seeding sensor sites: {e}")
        db.rollback()
    finally:
        # Close the session
        try:
            next(db_gen)
        except StopIteration:
            pass


def load_plan_if_not_exists(db, filename):
    """Load a plan from the given JSON file if no plan with the same name exists."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading plan file {filename}: {e}")
        return False

    # Check if a plan with this name already exists
    existing = db.query(ResponsePlan).filter(ResponsePlan.name == data['name']).first()
    if existing:
        print(f"Plan '{data['name']}' already exists. Skipping.")
        return False

    # Create the plan
    plan = ResponsePlan(
        name=data['name'],
        hazard_type=data['hazard_type'],
        region_name=data['region_name'],
        version=data.get('version', 1),
        plan_data=json.dumps(data.get('plan_data', {})),
        is_active=data.get('is_active', True),
    )
    db.add(plan)
    db.flush()

    # Add resources
    for r in data.get('resources', []):
        db.add(ResourcePosition(
            plan_id=plan.id,
            resource_type=r['resource_type'],
            quantity=r['quantity'],
            location=r['location'],
            capacity_per_hour=r.get('capacity_per_hour'),
            deployment_time_hours=r.get('deployment_time_hours'),
        ))

    # Add routes
    for rt in data.get('routes', []):
        db.add(EvacuationRoute(
            plan_id=plan.id,
            route_name=rt['route_name'],
            hazard_type=rt['hazard_type'],
            path=rt['path'],
            capacity_per_hour=rt.get('capacity_per_hour'),
            estimated_clearance_hours=rt.get('estimated_clearance_hours'),
            vulnerable_segments=json.dumps(rt.get('vulnerable_segments', [])),
        ))

    db.commit()
    print(f"Seeded plan: {plan.name}")
    return True


def load_sample_plans():
    """Load sample plans (US and India) if they don't already exist."""
    db_gen = get_db()
    db = next(db_gen)
    try:
        # Load the original US plan
        us_plan_file = _seed_file('sample_plan.json')
        if os.path.exists(us_plan_file):
            load_plan_if_not_exists(db, us_plan_file)

        # Load the India plan
        india_plan_file = _seed_file('sample_plan_india.json')
        if os.path.exists(india_plan_file):
            load_plan_if_not_exists(db, india_plan_file)
    except Exception as e:
        print(f"Error seeding sample plans: {e}")
        db.rollback()
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


if __name__ == "__main__":
    load_sensor_sites()
    load_sample_plans()