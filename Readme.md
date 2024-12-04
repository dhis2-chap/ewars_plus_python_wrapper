# Python Wrapper for Ewars Plus

This is a simple wrapper that makes the Ewars Plus model that is available through docker at `maquins/ewars_plumber_api:laos` compatible with CHAP.

This wrapper assumes that the ewars model api is running and available at port 3288. This command can be used:

```bash
docker run -it -dp 3288:3288 maquins/ewars_plumber_api:laos
```