/**
 * What a saved app workflow offers.
 *
 * Read from `/was/interface/api/app_exposure`, which answers each exposed input with the
 * type the node behind it declares and the value the workflow was saved with.
 */

import { fetchWithin } from "./request.mjs";

const LOG_NAME = "WASNodeSuite.AppExposure";

const ROUTE = "/was/interface/api/app_exposure";

/**
 * Ask what one saved app workflow offers.
 *
 * @param {string} name - The workflow's name, as the `app` widget stores it.
 * @returns {Promise<object|null>} `{app, inputs, results, panels, nodes, missing}`, or null
 *   when there is no such workflow or it could not be read.
 */
export async function fetchExposure(name) {
  if (!name) return null;
  try {
    const response = await fetchWithin(`${ROUTE}?app=${encodeURIComponent(name)}`, {
      cache: "no-store",
    });
    // 404 is the answer for a name no saved workflow matches.
    if (response?.status === 404) return null;
    if (!response?.ok) return null;
    return await response.json();
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to read what ${name} offers:`, error);
    return null;
  }
}
